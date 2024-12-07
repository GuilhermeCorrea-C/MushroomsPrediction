import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("Classificador de Cogumelos 🍄")
st.write("Este aplicativo prediz se um cogumelo é comestível ou venenoso com base nos atributos fornecidos.")

# Carregar o dataset e preparar os dados
def carregar_dados():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    columns = [
        "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
        "gill-attachment", "gill-spacing", "gill-size", "gill-color",
        "stalk-shape", "stalk-root", "stalk-surface-above-ring",
        "stalk-surface-below-ring", "stalk-color-above-ring",
        "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
        "ring-type", "spore-print-color", "population", "habitat"
    ]

    df = pd.read_csv(url, header=None, names=columns)

    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    label_encoders = {}
    for column in df.columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    return df, label_encoders

# Treinar o modelo
def treinar_modelo(df):
    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy

df, label_encoders = carregar_dados()
model, accuracy = treinar_modelo(df)

# Exibir a acurácia do modelo
st.write(f"### Acurácia do modelo: {accuracy:.2%}")

# Criar os inputs do usuário
st.sidebar.header("Insira os atributos do cogumelo")
user_inputs = {}

columns = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

for column in columns:
    options = label_encoders[column].classes_
    user_inputs[column] = st.sidebar.selectbox(f"{column}", options)

user_inputs_encoded = {col: label_encoders[col].transform([user_inputs[col]])[0] for col in columns}
user_data = pd.DataFrame([user_inputs_encoded])

# Legenda com possíveis valores de input
st.write("---")
st.subheader("Legenda dos Campos")
legend = """
- **cap-shape**: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s  
- **cap-surface**: fibrous=f, grooves=g, scaly=y, smooth=s  
- **cap-color**: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y  
- **bruises**: bruises=t, no=f  
- **odor**: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s  
- **gill-attachment**: attached=a, descending=d, free=f, notched=n  
- **gill-spacing**: close=c, crowded=w, distant=d  
- **gill-size**: broad=b, narrow=n  
- **gill-color**: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y  
- **stalk-shape**: enlarging=e, tapering=t  
- **stalk-root**: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?  
- **stalk-surface-above-ring**: fibrous=f, scaly=y, silky=k, smooth=s  
- **stalk-surface-below-ring**: fibrous=f, scaly=y, silky=k, smooth=s  
- **stalk-color-above-ring**: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y  
- **stalk-color-below-ring**: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y  
- **veil-type**: partial=p, universal=u  
- **veil-color**: brown=n, orange=o, white=w, yellow=y  
- **ring-number**: none=n, one=o, two=t  
- **ring-type**: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z  
- **spore-print-color**: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y  
- **population**: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y  
- **habitat**: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
"""
st.markdown(legend)

# Fazer a previsão
if st.button("Prever"):
    prediction = model.predict(user_data)
    predicted_class = label_encoders["class"].inverse_transform(prediction)

    if predicted_class[0] == "e":
        st.success("O cogumelo é comestível! 🍽️")
    else:
        st.error("Cuidado! O cogumelo é venenoso! ☠️")