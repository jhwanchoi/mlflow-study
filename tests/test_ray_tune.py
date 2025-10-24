"""
Quick test script for Ray Tune integration.

This script runs a minimal tuning test with 2 trials to verify the setup works.
"""

import logging
from src.tuning import tune_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    print("=" * 80)
    print("Ray Tune Integration Test")
    print("=" * 80)
    print("\nRunning 2 trials with 3 epochs each to test the setup...")
    print("This will take a few minutes.\n")

    try:
        # Run with grid search for testing
        from ray import tune as ray_tune

        search_space = {
            "learning_rate": ray_tune.grid_search([0.001, 0.005]),  # 2 values = 2 trials
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "epochs": 3,
        }

        # Use grid search without HyperOpt (grid_search is not compatible with HyperOpt)
        # NOTE: num_samples=1 because grid_search will create 2 trials (one for each learning_rate value)
        results = tune_model(
            num_samples=1,  # Grid search creates 2 trials automatically (2 learning_rate values)
            max_concurrent_trials=1,  # Run one at a time
            search_space=search_space,
            search_alg=None,  # Will use default BasicVariantGenerator for grid_search
        )

        print("\n" + "=" * 80)
        print("Test Complete!")
        print("=" * 80)
        print(f"\nTotal trials: {len(results)}")
        print("\nCheck MLflow UI at http://localhost:5001 to see the results!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
