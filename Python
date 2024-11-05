import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from datetime import datetime
import json
import uuid
import os
from flask import Flask, render_template, request

app = Flask(__name__)

class MedicalRecordSystem:
    def __init__(self):
        self.diseases = [
            "Gripe", "COVID-19", "Diabetes Tipo 2", "Hipertensão Arterial", "Asma",
            "Pneumonia", "Alergia Alimentar", "Doença de Alzheimer", "Depressão",
            "Anemia", "Câncer de Pulmão", "Doença de Crohn", "Fibromialgia",
            "Hipotireoidismo", "Esclerose Múltipla", "Síndrome do Intestino Irritável",
            "Artrite Reumatoide", "Doença Renal Crônica", "Enxaqueca", "Lúpus"
        ]

        self.symptoms = [
            "Febre", "dor de garganta", "tosse", "dor no corpo", "tosse seca",
            "falta de ar", "perda de olfato", "dor de cabeça", "Sede excessiva",
            "fome frequente", "perda de peso", "visão embaçada", "fadiga",
            "tontura", "palpitações", "Urticária", "inchaço", "dor abdominal", "diarreia"
        ]

        self.training_data = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        ])

        self.encoder = OneHotEncoder()
        self.labels_encoded = self.encoder.fit_transform(np.array(self.diseases).reshape(-1, 1)).toarray()
        self.model = self._build_model()
        self.records_dir = "medical_records"
        if not os.path.exists(self.records_dir):
            os.makedirs(self.records_dir)

    def _build_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(len(self.symptoms),)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(self.diseases), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.training_data, self.labels_encoded, epochs=100, batch_size=4, verbose=0)

    def symptoms_to_vector(self, symptom_list):
        return [1 if symptom in symptom_list else 0 for symptom in self.symptoms]

    def predict_disease(self, symptom_vector):
        if len(symptom_vector) < len(self.symptoms):
            symptom_vector.extend([0] * (len(self.symptoms) - len(symptom_vector)))
        symptom_vector = np.array(symptom_vector).reshape(1, -1)
        prediction = self.model.predict(symptom_vector)
        disease_index = np.argmax(prediction)
        confidence = float(prediction[0][disease_index]) * 100
        return self.diseases[disease_index], confidence

    def create_medical_record(self, patient_data, symptoms, observations=""):
        symptom_vector = self.symptoms_to_vector(symptoms)
        predicted_disease, confidence = self.predict_disease(symptom_vector)

        record = {
            "record_id": str(uuid.uuid4()),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient": patient_data,
            "consultation": {
                "symptoms": symptoms,
                "observations": observations,
                                "predicted_disease": predicted_disease,
                "confidence": f"{confidence:.2f}%"
            },
            "status": "active"
        }

        # Salva o registro no arquivo JSON
        filename = f"{self.records_dir}/{record['record_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=4)

        return record

    def get_medical_record(self, record_id):
        """Recupera um registro médico por ID"""
        try:
            with open(f"{self.records_dir}/{record_id}.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def update_medical_record(self, record_id, updates):
        """Atualiza um registro médico existente"""
        record = self.get_medical_record(record_id)
        if record:
            record.update(updates)
            with open(f"{self.records_dir}/{record_id}.json", 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=4)
            return record
        return None

    def get_patient_history(self, patient_id):
        """Recupera todos os registros médicos de um paciente específico"""
        patient_records = []
        for filename in os.listdir(self.records_dir):
            if filename.endswith('.json'):
                with open(f"{self.records_dir}/{filename}", 'r', encoding='utf-8') as f:
                    record = json.load(f)
                    if record['patient']['id'] == patient_id:
                        patient_records.append(record)
        return patient_records

# Inicializa o modelo e treina
mrs = MedicalRecordSystem()
mrs.train_model()

@app.route('/')
def index():
    return render_template("pagina.html")

@app.route('/create_record', methods=['POST'])
def create_record():
    patient_data = {
        "name": request.form['name'],
        "age": request.form['age'],
        "gender": request.form['gender'],
        "contact": request.form['contact']
    }
    symptoms = request.form.getlist('symptoms')
    observations = request.form['observations']

    record = mrs.create_medical_record(patient_data, symptoms, observations)
    return render_template("2pagina.html", record=record)

if __name__ == "__main__":
    app.run(debug=True)
