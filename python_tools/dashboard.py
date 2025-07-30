import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Configuración inicial de Streamlit
st.set_page_config(page_title="Dashboard de Resultados Fashion-MNIST Transformer", layout="wide")
st.title("📊 Dashboard de Visualización de Resultados - Fashion-MNIST Transformer")

# Clases de Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Función para cargar CSVs con manejo de errores
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Archivo no encontrado: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error cargando {file_path}: {e}")
        return None

# Cargar datos
training_history = load_data("training_history.csv")
test_metrics = load_data("test_metrics.csv")
predictions = load_data("predictions.csv")

# Sección 1: Historial de Entrenamiento
if training_history is not None:
    st.header("📈 Historial de Entrenamiento")
    col1, col2, col3 = st.columns(3)

    # Gráfico de Loss vs Epoch
    fig_loss = px.line(training_history, x="epoch", y="loss", title="Loss vs Epoch",
                       markers=True, labels={"loss": "Loss"})
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    col1.plotly_chart(fig_loss, use_container_width=True)

    # Gráfico de Accuracy vs Epoch
    fig_acc = px.line(training_history, x="epoch", y="accuracy", title="Accuracy vs Epoch",
                      markers=True, labels={"accuracy": "Accuracy (%)"})
    fig_acc.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy (%)")
    col2.plotly_chart(fig_acc, use_container_width=True)

    # Gráfico de Learning Rate vs Epoch
    fig_lr = px.line(training_history, x="epoch", y="learning_rate", title="Learning Rate vs Epoch",
                     markers=True, labels={"learning_rate": "Learning Rate"})
    fig_lr.update_layout(xaxis_title="Epoch", yaxis_title="Learning Rate (log)", yaxis_type="log")
    col3.plotly_chart(fig_lr, use_container_width=True)

    # Gráfico combinado
    st.subheader("Curvas Combinadas")
    fig_combined = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig_combined.add_trace(go.Scatter(x=training_history["epoch"], y=training_history["loss"], name="Loss", mode="lines+markers"))
    fig_combined.add_trace(go.Scatter(x=training_history["epoch"], y=training_history["accuracy"], name="Accuracy", mode="lines+markers", yaxis="y2"))
    fig_combined.update_layout(
        title="Loss y Accuracy vs Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis2=dict(title="Accuracy (%)", overlaying="y", side="right")
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    # Exportar datos
    if st.button("Descargar Training History CSV"):
        st.download_button(label="Descargar", data=training_history.to_csv(), file_name="training_history.csv", mime="text/csv")

else:
    st.warning("No se cargó training_history.csv. Verifica el archivo.")

# Sección 2: Métricas de Test
if test_metrics is not None:
    st.header("🎯 Métricas de Test")
    col1, col2 = st.columns(2)
    test_loss = test_metrics.get("test_loss", [0]).iloc[0]
    test_acc = test_metrics.get("test_accuracy", [0]).iloc[0]

    col1.metric("Test Loss", f"{test_loss:.5f}")
    col2.metric("Test Accuracy", f"{test_acc:.2f}%")

    # Gráfico de barras simple
    fig_test = go.Figure(data=[
        go.Bar(name="Test Loss", x=["Métricas"], y=[test_loss]),
        go.Bar(name="Test Accuracy", x=["Métricas"], y=[test_acc])
    ])
    fig_test.update_layout(title="Resumen de Métricas de Test", barmode="group")
    st.plotly_chart(fig_test, use_container_width=True)

    # Exportar
    if st.button("Descargar Test Metrics CSV"):
        st.download_button(label="Descargar", data=test_metrics.to_csv(), file_name="test_metrics.csv", mime="text/csv")

else:
    st.warning("No se cargó test_metrics.csv. Verifica el archivo.")

# Sección 3: Análisis de Predicciones
if predictions is not None:
    st.header("🔍 Análisis de Predicciones")
    true_labels = predictions["true_label"]
    pred_labels = predictions["predicted_label"]
    confidences = predictions.get("confidence", None)  # Opcional

    # Matriz de Confusión
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(true_labels, pred_labels)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="True", color="Count"),
                       x=class_names, y=class_names, color_continuous_scale="Blues")
    fig_cm.update_layout(title="Matriz de Confusión")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Accuracy por Clase
    st.subheader("Accuracy por Clase")
    class_acc = (cm.diagonal() / cm.sum(axis=1)) * 100
    fig_class_acc = px.bar(x=class_names, y=class_acc, labels={"y": "Accuracy (%)"},
                           title="Accuracy por Clase")
    fig_class_acc.update_layout(xaxis_title="Clase", yaxis_title="Accuracy (%)")
    st.plotly_chart(fig_class_acc, use_container_width=True)

    # Reporte de Clasificación
    st.subheader("Reporte de Clasificación")
    report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # Curvas ROC (multiclase)
    st.subheader("Curvas ROC")
    y_true_bin = label_binarize(true_labels, classes=range(10))
    
    # Simular probabilidades si no hay confidences (ajusta si tienes datos reales)
    if confidences is None:
        st.warning("No hay columna 'confidence'. Simulando probabilidades para ROC.")
        y_score = np.random.rand(len(true_labels), 10)
        for i, label in enumerate(true_labels):
            y_score[i, label] = np.max(y_score[i]) + 0.1  # Bias hacia label verdadera
        y_score = y_score / y_score.sum(axis=1, keepdims=True)
    else:
        # Usar confidences para aproximar (asumiendo una por predicción; ajusta si es necesario)
        y_score = np.zeros((len(true_labels), 10))
        for i, pred in enumerate(pred_labels):
            y_score[i, pred] = confidences.iloc[i]

    # Calcular ROC por clase
    fig_roc = go.Figure()
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{class_names[i]} (AUC={roc_auc:.2f})"))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="Curvas ROC Multiclase", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)

    # Filtro interactivo: Seleccionar clase para detalles
    selected_class = st.selectbox("Selecciona una clase para detalles", class_names)
    class_idx = class_names.index(selected_class)
    class_true = true_labels == class_idx
    class_pred = pred_labels == class_idx
    st.write(f"Predicciones para {selected_class}:")
    st.write(f"- Verdaderos Positivos: {np.sum(class_true & class_pred)}")
    st.write(f"- Falsos Positivos: {np.sum(~class_true & class_pred)}")
    st.write(f"- Falsos Negativos: {np.sum(class_true & ~class_pred)}")

    # Exportar
    if st.button("Descargar Predictions CSV"):
        st.download_button(label="Descargar", data=predictions.to_csv(), file_name="predictions.csv", mime="text/csv")

else:
    st.warning("No se cargó predictions.csv. Verifica el archivo.")

st.info("Dashboard creado por Grok AI. Ejecuta con Streamlit para interactividad completa.") 