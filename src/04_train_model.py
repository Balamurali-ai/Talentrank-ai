import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def main():
    print("Loading data...")
    df = pd.read_csv('models/weak_label_training_data.csv')
    
    features = [
        'semantic_sim', 
        'skill_overlap_score', 
        'experience_match_score', 
        'education_match_score', 
        'tfidf_sim'
    ]
    
    print("Feature columns used for training (X):")
    for f in features:
        print(f"  - {f}")
        
    X = df[features]
    y = df['label']
    
    categories = df['mapped_category']
    
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, categories, test_size=0.2, stratify=y, random_state=42
    )
    
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    print("\nTraining RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    print("Training XGBClassifier...")
    xgb = XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    def evaluate_model(model, name):
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)
        
        print(f"\n--- {name} Evaluation ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        print(f"\n{name} Feature Importances:")
        importances = model.feature_importances_
        for f, imp in zip(features, importances):
            print(f"  {f}: {imp:.4f}")
            
        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }

    rf_metrics = evaluate_model(rf, "RandomForest")
    xgb_metrics = evaluate_model(xgb, "XGBoost")
    
    def per_category_recall(model, name):
        print(f"\n--- Per-Category Recall ({name}) ---")
        test_df = pd.DataFrame({'y_true': y_test, 'y_pred': model.predict(X_test), 'category': cat_test})
        for cat in sorted(test_df['category'].unique()):
            cat_data = test_df[test_df['category'] == cat]
            pos_data = cat_data[cat_data['y_true'] == 1]
            if len(pos_data) > 0:
                rec = recall_score(pos_data['y_true'], pos_data['y_pred'], zero_division=0)
                print(f"  {cat}: {rec:.4f} (from {len(pos_data)} positive examples)")
            else:
                print(f"  {cat}: N/A (0 positive examples in test set)")

    per_category_recall(rf, "RandomForest")
    per_category_recall(xgb, "XGBoost")
    
    best_model_name = "RandomForest"
    best_model = rf
    best_metrics = rf_metrics
    
    if xgb_metrics['f1'] > rf_metrics['f1']:
        best_model_name = "XGBoost"
        best_model = xgb
        best_metrics = xgb_metrics
        
    print(f"\nSelecting {best_model_name} as the final model based on F1 Score ({best_metrics['f1']:.4f}).")
    
    model_path = 'models/candidate_scorer.pkl'
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")
    
    metadata = {
        'chosen_algorithm': best_model_name,
        'metrics': best_metrics,
        'features': features,
        'known_limitations': "BPO category is severely underrepresented (22 resumes total in source data, only 9 distinct ones labeled as positive), leading to poor generalization/recall for this category."
    }
    
    metadata_path = 'models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()
