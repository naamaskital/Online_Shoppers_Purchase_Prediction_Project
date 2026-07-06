## שיפור המודל המוביל

לאחר השוואת המודלים הראשונית, Random Forest נמצא כמודל המוביל לפי מדד F1. כדי לשפר את המודל, בוצע Hyperparameter Tuning באמצעות GridSearchCV עם Stratified K-Fold Cross Validation. החיפוש בוצע לפי מדד F1, משום שהדאטה אינו מאוזן ויש חשיבות לאיזון בין Precision ו-Recall.

הפרמטרים הטובים ביותר שנמצאו הם:

- classifier__class_weight: balanced
- classifier__max_depth: 20
- classifier__max_features: sqrt
- classifier__min_samples_leaf: 1
- classifier__n_estimators: 300

ציון ה-F1 הממוצע הטוב ביותר ב-Cross Validation היה 0.6871.

בנוסף, בוצע Threshold Tuning על סט Validation. במקום להשתמש תמיד בסף ברירת המחדל 0.5, נבדקו ספים שונים, ונבחר הסף שנתן את ערך ה-F1 הגבוה ביותר על סט ה-Validation.

הסף הטוב ביותר שנמצא הוא 0.47.

תוצאות המודל המשופר:

