from flask import Flask, request, jsonifyfrom 
flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilita CORS en todas las rutas para simplificar el acceso desde cualquier origen

# Cargar el modelo de regresión logística
model_path = 'logistic_regression_model_cardio.pkl'  # Asegúrate de ajustar la ruta según tu configuración
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()

        # Validar y convertir los datos
        required_fields = [
            'pasos_diarios', 'pulsaciones_diarias', 'presion_sistolica',
            'presion_diastolica', 'peso', 'edad', 'antecedentes_familiares',
            'altura', 'imc'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'El campo {field} es requerido.'}), 400

        try:
            pasos_diarios = min(round(float(data['pasos_diarios'])), 8000)
            pulsaciones_diarias = float(data['pulsaciones_diarias'])
            presion_sistolica = float(data['presion_sistolica'])
            presion_diastolica = float(data['presion_diastolica'])
            peso = float(data['peso'])
            edad = int(data['edad'])
            antecedentes_familiares = int(data['antecedentes_familiares'])
            altura = float(data['altura'])
            imc = float(data['imc'])
        except ValueError as e:
            return jsonify({'error': f'Error en la conversión de datos: {str(e)}'}), 400

        # Extracción de características
        features = [pasos_diarios, pulsaciones_diarias, presion_sistolica,
                    presion_diastolica, peso, edad,
                    antecedentes_familiares, altura, imc]
         # Conversión a forma adecuada para el modelo (1 fila, varias columnas)
        features = np.array(features).reshape(1, -1)

        # Predicción de la probabilidad de ataque al corazón
        prediction_probability = model.predict_proba(features)

        # Obtener la probabilidad de ataque al corazón (clase 1)
        attack_probability = prediction_probability[0][1]

        # Retornar la probabilidad en formato JSON
        return jsonify({'probabilidad_ataque_cardiaco': attack_probability}), 200
    except Exception as e:
        # Imprimir el error para depuración
        print("Error:", str(e))
        # Retornar el error en formato JSON con el código de estado HTTP 500
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)