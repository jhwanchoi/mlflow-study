# S3 Bucket for MLflow Artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = var.s3_bucket_name

  tags = {
    Name        = var.s3_bucket_name
    Purpose     = "MLflow Artifacts Storage"
    Environment = var.environment
  }
}

# Enable versioning for artifact protection
resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy to optimize storage costs
resource "aws_s3_bucket_lifecycle_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    # Transition old artifacts to cheaper storage
    transition {
      days          = 90
      storage_class = "GLACIER_IR"  # Glacier Instant Retrieval
    }

    transition {
      days          = 180
      storage_class = "DEEP_ARCHIVE"
    }

    # Clean up old non-current versions
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER_IR"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }

    # Clean up incomplete multipart uploads
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "delete-old-model-artifacts"
    status = "Enabled"

    # Optional: Auto-delete artifacts older than 2 years
    # Remove this rule if you want to keep all artifacts
    filter {
      prefix = "artifacts/"
    }

    expiration {
      days = 730  # 2 years
    }
  }
}

# CORS configuration for direct browser uploads (if needed)
resource "aws_s3_bucket_cors_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "DELETE", "HEAD"]
    allowed_origins = var.s3_cors_allowed_origins
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# Bucket policy for MLflow service account access
resource "aws_s3_bucket_policy" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowMLflowServiceAccount"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.mlflow_s3_access.arn
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        Sid    = "DenyInsecureTransport"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# CloudWatch metrics for S3 bucket monitoring
resource "aws_cloudwatch_metric_alarm" "s3_bucket_size" {
  alarm_name          = "${var.s3_bucket_name}-size-warning"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "BucketSizeBytes"
  namespace           = "AWS/S3"
  period              = "86400"  # 24 hours
  statistic           = "Average"
  threshold           = var.s3_size_alarm_threshold  # bytes (e.g., 100GB = 107374182400)
  alarm_description   = "This metric monitors S3 bucket size"
  alarm_actions       = []  # Add SNS topic ARN for notifications

  dimensions = {
    BucketName  = aws_s3_bucket.mlflow_artifacts.id
    StorageType = "StandardStorage"
  }

  tags = {
    Name = "${var.s3_bucket_name}-size-warning"
  }
}

# S3 Bucket for Terraform State (optional, but recommended)
# This should be created manually first or in a separate bootstrap configuration
# resource "aws_s3_bucket" "terraform_state" {
#   bucket = "${var.project_name}-terraform-state"
#
#   tags = {
#     Name    = "${var.project_name}-terraform-state"
#     Purpose = "Terraform State Storage"
#   }
# }
#
# resource "aws_s3_bucket_versioning" "terraform_state" {
#   bucket = aws_s3_bucket.terraform_state.id
#
#   versioning_configuration {
#     status = "Enabled"
#   }
# }
#
# resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
#   bucket = aws_s3_bucket.terraform_state.id
#
#   rule {
#     apply_server_side_encryption_by_default {
#       sse_algorithm = "AES256"
#     }
#   }
# }
#
# resource "aws_s3_bucket_public_access_block" "terraform_state" {
#   bucket = aws_s3_bucket.terraform_state.id
#
#   block_public_acls       = true
#   block_public_policy     = true
#   ignore_public_acls      = true
#   restrict_public_buckets = true
# }
#
# # DynamoDB table for Terraform state locking
# resource "aws_dynamodb_table" "terraform_locks" {
#   name         = "${var.project_name}-terraform-locks"
#   billing_mode = "PAY_PER_REQUEST"
#   hash_key     = "LockID"
#
#   attribute {
#     name = "LockID"
#     type = "S"
#   }
#
#   tags = {
#     Name    = "${var.project_name}-terraform-locks"
#     Purpose = "Terraform State Locking"
#   }
# }
