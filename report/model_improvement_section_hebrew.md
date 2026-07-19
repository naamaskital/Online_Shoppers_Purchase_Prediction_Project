## שיפור המודל המוביל

לאחר השוואת המודלים הראשונית, Random Forest נמצא כמודל המוביל לפי מדד F1. כדי לשפר את המודל, בוצע Hyperparameter Tuning באמצעות GridSearchCV עם Stratified K-Fold Cross Validation. החיפוש בוצע לפי מדד F1, משום שהדאטה אינו מאוזן ויש חשיבות לאיזון בין Precision ו-Recall.

הפרמטרים הטובים ביותר שנמצאו הם:

- classifier__class_weight: balanced
- classifier__max_depth: None
- classifier__max_features: sqrt
- classifier__min_samples_leaf: 1
- classifier__n_estimators: 100

ציון ה-F1 הממוצע הטוב ביותר ב-Cross Validation היה 0.6830.

בנוסף, בוצע Threshold Tuning על סט Validation. במקום להשתמש תמיד בסף ברירת המחדל 0.5, נבדקו ספים שונים, ונבחר הסף שנתן את ערך ה-F1 הגבוה ביותר על סט ה-Validation.

הסף הטוב ביותר שנמצא הוא 0.45.

תוצאות המודל המשופר:

| Model                                |   Accuracy |   Precision |   Recall |       F1 |   ROC_AUC |
|:-------------------------------------|-----------:|------------:|---------:|---------:|----------:|
| Baseline Random Forest               |   0.886861 |    0.623501 | 0.680628 | 0.650814 |  0.91492  |
| Tuned Random Forest - threshold 0.5  |   0.884428 |    0.614657 | 0.680628 | 0.645963 |  0.913915 |
| Tuned Random Forest - threshold 0.45 |   0.879562 |    0.591793 | 0.717277 | 0.648521 |  0.913915 |

שיפור זה הופך את תהליך בניית המודל למקצועי יותר, משום שהוא אינו מסתפק בהרצת מודל בודד עם פרמטרים קבועים, אלא בודק שילובים שונים של פרמטרים ובוחר סף החלטה שמתאים טוב יותר לבעיה.
