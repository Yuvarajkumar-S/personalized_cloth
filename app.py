from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from random_forest_recommender import RandomForestRecommender
import time

app = Flask(__name__)

# Initialize recommender
recommender = RandomForestRecommender()

# Load trained models
model_path = 'models/'
models_loaded = False

if os.path.exists(model_path):
    models_loaded = recommender.load_models(model_path)
    
if not models_loaded:
    print("⚠️ No trained models found. Please run train_model.py first")
    print("📊 Using fallback mode (training on the fly)")
    try:
        recommender.train('data/recommendations.csv')
        recommender.save_models('models/')
        models_loaded = True
    except:
        print("❌ Could not train models. Please check your data file.")

# Load dataset for dropdown options
def load_attribute_options():
    try:
        df = pd.read_csv('data/recommendations.csv')
        options = {
            'hair_colors': sorted(df['Hair Color'].unique()),
            'eye_colors': sorted(df['Eye Color'].unique()),
            'skin_tones': sorted(df['Skin Tone'].unique()),
            'under_tones': sorted(df['Under Tone'].unique()),
            'torso_lengths': sorted(df['Torso length'].unique()),
            'body_proportions': sorted(df['Body Proportion'].unique())
        }
        return options
    except:
        return {
            'hair_colors': ['Black', 'Brown', 'Blonde', 'Red'],
            'eye_colors': ['Brown', 'Blue', 'Green', 'Hazel'],
            'skin_tones': ['Fair', 'Medium', 'Olive', 'Dark'],
            'under_tones': ['Warm', 'Cool', 'Neutral'],
            'torso_lengths': ['Short', 'Balanced', 'Long'],
            'body_proportions': ['Hourglass', 'Rectangle', 'Apple', 'Pear']
        }

options = load_attribute_options()

@app.route('/')
def index():
    return render_template('index.html', options=options)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        start_time = time.time()
        
        user_attributes = {
            'Hair Color': request.form.get('hair_color', ''),
            'Eye Color': request.form.get('eye_color', ''),
            'Skin Tone': request.form.get('skin_tone', ''),
            'Under Tone': request.form.get('under_tone', ''),
            'Torso length': request.form.get('torso_length', ''),
            'Body Proportion': request.form.get('body_proportion', '')
        }
        
        recommendations = recommender.predict(user_attributes)
        
        # ========== DYNAMIC IMAGES BASED ON USER INPUT ==========
        
        # Get user attributes for dynamic image selection
        hair = user_attributes.get('Hair Color', 'Black')
        body_type = user_attributes.get('Body Proportion', 'Hourglass')
        undertone = user_attributes.get('Under Tone', 'Warm')
        style = recommendations['fitting_style']
        jewelry = recommendations['jewelry_metal']
        
        # Style based image mapping
        style_images = {
            'Tailored Fit': 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=500&fit=crop',
            'A-Line Bottoms': 'https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=400&h=500&fit=crop',
            'Wraps and Empire': 'https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400&h=500&fit=crop',
            'Empire Waist': 'https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=400&h=500&fit=crop',
            'Balanced Fit': 'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=400&h=500&fit=crop',
            'Defined Waist': 'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=400&h=500&fit=crop',
            'default': 'https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=400&h=500&fit=crop'
        }
        
        # Body type based image mapping
        body_images = {
            'Hourglass': 'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=400&h=500&fit=crop',
            'Rectangle': 'https://images.unsplash.com/photo-1483985988355-763728e1935b?w=400&h=500&fit=crop',
            'Apple': 'https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400&h=500&fit=crop',
            'Pear': 'https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=400&h=500&fit=crop',
            'Inverted Triangle': 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=500&fit=crop',
            'default': 'https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=400&h=500&fit=crop'
        }
        
        # Undertone based images
        undertone_images = {
            'Warm': 'https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=400&h=500&fit=crop',
            'Cool': 'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=400&h=500&fit=crop',
            'Neutral': 'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=400&h=500&fit=crop',
            'default': 'https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=400&h=500&fit=crop'
        }
        
        # Select images based on user attributes
        image1 = style_images.get(style, style_images['default'])
        image2 = body_images.get(body_type, body_images['default'])
        image3 = undertone_images.get(undertone, undertone_images['default'])
        
        outfit_images = [
            {
                'name': f'{style} Outfit',
                'description': f'Perfect {style.lower()} that flatters your {body_type} figure',
                'image_url': image1,
                'colors': recommendations['recommended_colors'][:3],
                'price': '$89.99'
            },
            {
                'name': f'{body_type} Collection',
                'description': f'Specially designed for {body_type} body type with {undertone} undertone',
                'image_url': image2,
                'colors': recommendations['recommended_colors'][:2],
                'price': '$49.99'
            },
            {
                'name': f'{jewelry} Essentials',
                'description': f'Complete your {undertone} look with {jewelry.lower()} accessories',
                'image_url': image3,
                'colors': [jewelry],
                'price': '$29.99'
            }
        ]
        
        return render_template('recommendations.html',
                             recommendations=recommendations,
                             user_attributes=user_attributes,
                             outfit_images=outfit_images,
                             prediction_time=round(time.time() - start_time, 2))
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.get_json()
        user_attributes = {
            'Hair Color': data.get('hair_color', ''),
            'Eye Color': data.get('eye_color', ''),
            'Skin Tone': data.get('skin_tone', ''),
            'Under Tone': data.get('under_tone', ''),
            'Torso length': data.get('torso_length', ''),
            'Body Proportion': data.get('body_proportion', '')
        }
        recommendations = recommender.predict(user_attributes)
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("👗 CLOTHING RECOMMENDATION SYSTEM")
    print("="*60)
    print("🚀 Starting web server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)