from random_forest_recommender import RandomForestRecommender
import os

def main():
    print("\n" + "="*70)
    print("🎯 RANDOM FOREST CLOTHING RECOMMENDER - TRAINING")
    print("="*70 + "\n")
    
    recommender = RandomForestRecommender(n_estimators=100, max_depth=10, random_state=42)
    data_path = 'recommendations.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        print("Please place your recommendations.csv file in the 'data' folder")
        return
    
    recommender.train(data_path)
    recommender.save_models('models/')
    
    # Test prediction
    print("\n📊 Testing prediction...")
    test_user = {
        'Hair Color': 'Black',
        'Eye Color': 'Brown',
        'Skin Tone': 'Medium',
        'Under Tone': 'Warm',
        'Torso length': 'Balanced',
        'Body Proportion': 'Hourglass'
    }
    result = recommender.predict(test_user)
    print(f"✅ Test prediction successful!")
    print(f"   Recommended Colors: {', '.join(result['recommended_colors'][:3])}")
    
    print("\n✅ Training completed successfully!")
    print("🚀 You can now run: python app.py")

if __name__ == "__main__":
    main()