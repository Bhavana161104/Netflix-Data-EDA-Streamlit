from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def run_model(df):
    df = df.dropna(subset=['type'])

    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])

    X = df[['release_year']]
    y = df['type_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return acc
