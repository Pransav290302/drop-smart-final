

"""Test ML Pipeline Service - Diagnose Backend Issues"""
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

print("=" * 60)
print("DROPSMART ML PIPELINE DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Import Check
print("\n[1/5] Testing imports...")
try:
    from backend.app.services.pipeline_service import pipeline_service
    print("    ✅ Pipeline service imported successfully")
except Exception as e:
    print(f"    ❌ Failed to import pipeline_service: {e}")
    sys.exit(1)

# Test 2: Model Health Check
print("\n[2/5] Checking model health...")
try:
    health = pipeline_service.health_check()
    print("    Model Status:")
    for model_name, is_loaded in health.items():
        status = "✅ LOADED" if is_loaded else "❌ NOT LOADED"
        print(f"      - {model_name}: {status}")
    
    if not any(health.values()):
        print("\n    ⚠️  WARNING: No models are loaded!")
        print("    ⚠️  You need to train models first.")
        print("    ⚠️  Run: python -m backend.app.ml.scripts.train_models")
except Exception as e:
    print(f"    ❌ Health check failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Config Check
print("\n[3/5] Checking configuration...")
try:
    from backend.app.core.config import settings
    print(f"    Models directory: {settings.models_dir}")
    
    models_path = Path(settings.models_dir)
    if models_path.exists():
        print(f"    ✅ Models directory exists")
        # List model files
        model_files = list(models_path.rglob("*.pkl"))
        if model_files:
            print(f"    Found {len(model_files)} model files:")
            for mf in model_files:
                print(f"      - {mf.relative_to(models_path)}")
        else:
            print(f"    ⚠️  No .pkl model files found in {models_path}")
    else:
        print(f"    ❌ Models directory does NOT exist: {models_path}")
except Exception as e:
    print(f"    ❌ Config check failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Create Test Data
print("\n[4/5] Creating test product data...")
try:
    import pandas as pd
    
    test_df = pd.DataFrame({
        "sku": ["TEST001"],
        "product_name": ["Test Product"],
        "cost": [10.0],
        "price": [20.0],
        "shipping_cost": [5.0],
        "lead_time_days": [7],
        "availability": ["in_stock"],
        "weight_kg": [1.0],
        "length_cm": [10.0],
        "width_cm": [10.0],
        "height_cm": [10.0],
        "stock": [100],
        "inventory": [100],
        "quantity": [100],
        "demand": [10],
        "past_sales": [50],
        "supplier_reliability_score": [0.8],
        "duties": [2.0],
    })
    
    print(f"    ✅ Created test DataFrame with {len(test_df)} products")
    print(f"    Columns: {list(test_df.columns)}")
    
except Exception as e:
    print(f"    ❌ Failed to create test data: {e}")
    sys.exit(1)

# Test 5: Run Pipeline
print("\n[5/5] Testing pipeline processing...")
try:
    results = pipeline_service.process_products(test_df)
    
    print(f"    ✅ Pipeline processing successful!")
    print(f"\n    Results for '{results[0]['sku']}':")
    print(f"      - Product: {results[0]['product_name']}")
    print(f"      - Viability Score: {results[0]['viability_score']:.2%}")
    print(f"      - Viability Class: {results[0]['viability_class']}")
    print(f"      - Current Price: ${results[0]['current_price']:.2f}")
    print(f"      - Recommended Price: ${results[0]['recommended_price']:.2f}")
    print(f"      - Margin: {results[0]['margin_percent']:.1f}%")
    print(f"      - Stockout Risk: {results[0]['stockout_risk_level']} ({results[0]['stockout_risk_score']:.2%})")
    if results[0].get('cluster_id') is not None:
        print(f"      - Cluster ID: {results[0]['cluster_id']}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Pipeline is working correctly!")
    print("=" * 60)
    
except Exception as e:
    print(f"    ❌ Pipeline processing failed!")
    print(f"\n    Error: {e}")
    print("\n    Full traceback:")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("\nLikely causes:")
    print("  1. Models not trained yet")
    print("     → Run: cd backend && python -m app.ml.scripts.train_models")
    print("\n  2. Missing model files in models/ directory")
    print("     → Check that models/viability/model.pkl exists")
    print("\n  3. Missing required columns in data")
    print("     → Check column names match schema")
    print("\n  4. Import path issues")
    print("     → Verify backend module structure")
    print("=" * 60)
