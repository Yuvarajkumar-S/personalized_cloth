import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class RandomForestRecommender:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_encoders = {}
        self.models = {}
        self.mlb = {}
        
        self.feature_columns = ['Hair Color', 'Eye Color', 'Skin Tone', 
                                 'Under Tone', 'Torso length', 'Body Proportion']
        
        self.single_output_cols = [
            'Recommended Fitting Style', 'Recommended Jewelry Metal',
            'Recommended Shoes', 'Recommended Clothing Color Wheel Region',
            'Fabric Nature', 'Do Exaggerate', "Don't Exaggerate"
        ]
        
        self.multi_output_cols = [
            'Recommended Clothing Colors', 'Avoid Clothing Colors',
            'Recommended Materials', 'Recommended Patterns'
        ]
        
        self.all_output_cols = self.single_output_cols + self.multi_output_cols
    
    def encode_features(self, df):
        df_encoded = df.copy()
        for col in self.feature_columns:
            if col not in self.feature_encoders:
                self.feature_encoders[col] = LabelEncoder()
                self.feature_encoders[col].fit(df[col].astype(str))
            df_encoded[col] = self.feature_encoders[col].transform(df[col].astype(str))
        return df_encoded[self.feature_columns]
    
    def encode_single_output(self, df, col):
        if col not in self.mlb:
            self.mlb[col] = LabelEncoder()
            self.mlb[col].fit(df[col].astype(str))
        return self.mlb[col].transform(df[col].astype(str))
    
    def encode_multi_output(self, df, col):
        value_lists = df[col].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) and str(x) != 'nan' else [])
        if col not in self.mlb:
            self.mlb[col] = MultiLabelBinarizer()
            all_values = []
            for lst in value_lists:
                all_values.extend(lst)
            self.mlb[col].fit([list(set(all_values))])
        return self.mlb[col].transform(value_lists)
    
    def train(self, data_path):
        print("="*60)
        print("🎯 TRAINING RANDOM FOREST RECOMMENDER")
        print("="*60)
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records")
        X = self.encode_features(df)
        print(f"✅ Features encoded")
        
        for col in self.all_output_cols:
            print(f"📊 Training: {col[:40]}...", end=" ", flush=True)
            if col in self.single_output_cols:
                y = self.encode_single_output(df, col)
            else:
                y = self.encode_multi_output(df, col)
            
            rf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                        random_state=self.random_state, n_jobs=-1)
            rf.fit(X, y)
            self.models[col] = rf
            print("✅")
        
        print("="*60)
        print(f"🏆 Training complete! {len(self.models)} models ready")
        print("="*60)
        return True
    
    def predict(self, user_attributes):
        user_df = pd.DataFrame([user_attributes])
        X = self.encode_features(user_df)
        recommendations = {}
        
        for col in self.all_output_cols:
            if col in self.models:
                prediction = self.models[col].predict(X)[0]
                if col in self.single_output_cols:
                    decoded = self.mlb[col].inverse_transform([prediction])[0]
                else:
                    decoded = self.mlb[col].inverse_transform(prediction.reshape(1, -1))[0]
                if col in self.multi_output_cols:
                    recommendations[col] = list(decoded) if isinstance(decoded, (list, tuple)) else [decoded]
                else:
                    recommendations[col] = decoded
        
        return {
            "recommended_colors": recommendations.get('Recommended Clothing Colors', ['Earth Tones', 'Olive', 'Coral']),
            "avoid_colors": recommendations.get('Avoid Clothing Colors', ['Cool Blue', 'Icy Gray']),
            "fitting_style": recommendations.get('Recommended Fitting Style', 'Tailored Fit'),
            "materials": recommendations.get('Recommended Materials', ['Cotton', 'Linen']),
            "patterns": recommendations.get('Recommended Patterns', ['Solid', 'Subtle Prints']),
            "jewelry_metal": recommendations.get('Recommended Jewelry Metal', 'Gold'),
            "shoes": recommendations.get('Recommended Shoes', 'Low Heels'),
            "color_wheel_region": recommendations.get('Recommended Clothing Color Wheel Region', 'Warm Colors'),
            "fabric_nature": recommendations.get('Fabric Nature', 'Stretchy'),
            "exaggerate": recommendations.get('Do Exaggerate', 'Highlight waistline'),
            "dont_exaggerate": recommendations.get("Don't Exaggerate", 'Dont exaggerate straight lines')
        }
    
    def save_models(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.feature_encoders, f'{path}/feature_encoders.pkl')
        joblib.dump(self.mlb, f'{path}/label_encoders.pkl')
        for col, model in self.models.items():
            safe_name = col.replace(' ', '_').replace('/', '_')
            joblib.dump(model, f'{path}/model_{safe_name}.pkl')
        print(f"✅ Models saved to {path}")
    
    def load_models(self, path='models/'):
        if not os.path.exists(path):
            return False
        try:
            self.feature_encoders = joblib.load(f'{path}/feature_encoders.pkl')
            self.mlb = joblib.load(f'{path}/label_encoders.pkl')
            for col in self.all_output_cols:
                safe_name = col.replace(' ', '_').replace('/', '_')
                model_path = f'{path}/model_{safe_name}.pkl'
                if os.path.exists(model_path):
                    self.models[col] = joblib.load(model_path)
            print(f"✅ {len(self.models)} models loaded")
            return True
        except:
            return False