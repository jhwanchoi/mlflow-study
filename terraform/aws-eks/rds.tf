# RDS Subnet Group
resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project_name}-mlflow-db-subnet"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-mlflow-db-subnet"
  }
}

# Random password for RDS
resource "random_password" "rds_password" {
  length  = 32
  special = true
  # Exclude characters that might cause issues in connection strings
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# Store RDS password in AWS Secrets Manager
resource "aws_secretsmanager_secret" "rds_password" {
  name                    = "${var.project_name}-mlflow-db-password"
  description             = "Password for MLflow RDS PostgreSQL database"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.project_name}-mlflow-db-password"
  }
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  secret_id = aws_secretsmanager_secret.rds_password.id
  secret_string = jsonencode({
    username = var.rds_username
    password = random_password.rds_password.result
    engine   = "postgres"
    host     = aws_db_instance.mlflow.address
    port     = aws_db_instance.mlflow.port
    dbname   = var.rds_database_name
  })
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "mlflow" {
  identifier = var.rds_identifier

  # Engine configuration
  engine               = "postgres"
  engine_version       = var.rds_engine_version
  instance_class       = var.rds_instance_class
  allocated_storage    = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type         = "gp3"
  storage_encrypted    = true

  # Database configuration
  db_name  = var.rds_database_name
  username = var.rds_username
  password = random_password.rds_password.result
  port     = 5432

  # Network configuration
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # High availability
  multi_az = var.environment == "prod" ? true : false

  # Backup configuration
  backup_retention_period = var.rds_backup_retention_period
  backup_window          = "03:00-04:00"  # UTC
  maintenance_window     = "mon:04:00-mon:05:00"  # UTC

  # Deletion protection
  deletion_protection = var.environment == "prod" ? true : false
  skip_final_snapshot = var.environment == "dev" ? true : false
  final_snapshot_identifier = var.environment == "dev" ? null : "${var.rds_identifier}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  # Performance Insights
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  performance_insights_enabled    = var.environment == "prod" ? true : false
  performance_insights_retention_period = var.environment == "prod" ? 7 : null

  # Auto minor version upgrade
  auto_minor_version_upgrade = true

  # Parameter group for PostgreSQL optimization
  parameter_group_name = aws_db_parameter_group.mlflow.name

  tags = {
    Name = var.rds_identifier
  }

  lifecycle {
    ignore_changes = [
      final_snapshot_identifier  # Prevent Terraform from trying to update this on each apply
    ]
  }
}

# RDS Parameter Group for MLflow optimization
resource "aws_db_parameter_group" "mlflow" {
  name   = "${var.project_name}-mlflow-pg"
  family = "postgres15"

  description = "Custom parameter group for MLflow PostgreSQL"

  # Optimize for MLflow workload (many small transactions)
  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory/32768}"  # 25% of instance memory
  }

  parameter {
    name  = "effective_cache_size"
    value = "{DBInstanceClassMemory/16384}"  # 75% of instance memory
  }

  parameter {
    name  = "maintenance_work_mem"
    value = "262144"  # 256MB
  }

  parameter {
    name  = "checkpoint_completion_target"
    value = "0.9"
  }

  parameter {
    name  = "wal_buffers"
    value = "16384"  # 16MB
  }

  parameter {
    name  = "default_statistics_target"
    value = "100"
  }

  parameter {
    name  = "random_page_cost"
    value = "1.1"  # Optimized for SSD
  }

  parameter {
    name  = "effective_io_concurrency"
    value = "200"  # SSD
  }

  parameter {
    name  = "work_mem"
    value = "10485"  # 10MB per operation
  }

  parameter {
    name  = "min_wal_size"
    value = "1024"  # 1GB
  }

  parameter {
    name  = "max_wal_size"
    value = "4096"  # 4GB
  }

  # Connection settings
  parameter {
    name  = "max_connections"
    value = "100"
  }

  tags = {
    Name = "${var.project_name}-mlflow-pg"
  }
}

# CloudWatch alarms for RDS monitoring
resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  alarm_name          = "${var.rds_identifier}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS CPU utilization"
  alarm_actions       = []  # Add SNS topic ARN for notifications

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow.id
  }

  tags = {
    Name = "${var.rds_identifier}-high-cpu"
  }
}

resource "aws_cloudwatch_metric_alarm" "rds_storage" {
  alarm_name          = "${var.rds_identifier}-low-storage"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "5000000000"  # 5GB in bytes
  alarm_description   = "This metric monitors RDS free storage space"
  alarm_actions       = []  # Add SNS topic ARN for notifications

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow.id
  }

  tags = {
    Name = "${var.rds_identifier}-low-storage"
  }
}

resource "aws_cloudwatch_metric_alarm" "rds_connections" {
  alarm_name          = "${var.rds_identifier}-high-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"  # 80% of max_connections
  alarm_description   = "This metric monitors RDS database connections"
  alarm_actions       = []  # Add SNS topic ARN for notifications

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.mlflow.id
  }

  tags = {
    Name = "${var.rds_identifier}-high-connections"
  }
}
