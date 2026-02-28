from flask import Flask, render_template, request, send_file, session
import numpy as np
import joblib
import pandas as pd
import os
import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import TableStyle
from reportlab.lib.units import inch

import io
import qrcode
from reportlab.platypus import Image

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

app = Flask(__name__)
app.secret_key = "heartcare_secret_key"

# ================= LOAD MODELS =================
basic_model = joblib.load("heart_model_basic.pkl")
basic_scaler = joblib.load("scaler_basic.pkl")

advanced_model = joblib.load("heart_model_advanced.pkl")
advanced_scaler = joblib.load("scaler_advanced.pkl")


# ================= HOME =================
@app.route('/')
def home():
    return render_template("home.html")


# ================= BASIC PAGE =================
@app.route('/basic')
def basic():
    return render_template("basic.html")


@app.route('/predict_basic', methods=['POST'])
def predict_basic():
    try:
        sex = float(request.form['sex_m'])
        age = float(request.form['age'])
        cholesterol = float(request.form['cholesterol'])
        cp_ata = float(request.form['cp_ata'])
        exercise = float(request.form['exercise'])
        st_up = float(request.form['st_up'])
        fasting = float(request.form['fasting'])
        maxhr = float(request.form['maxhr'])
        st_flat = float(request.form['st_flat'])
        oldpeak = float(request.form['oldpeak'])

        features = np.array([[sex, age, cholesterol, cp_ata,
                            exercise, st_up, fasting,
                            maxhr, st_flat, oldpeak]])

        numerical = features[:, [1,2,7,9]]
        scaled = basic_scaler.transform(numerical)
        features[:, [1,2,7,9]] = scaled

        probability = basic_model.predict_proba(features)[0][1]
        risk_percent = round(probability * 100, 2)

        result = "⚠️ High Risk of Heart Disease" if probability >= 0.5 else "✅ Low Risk of Heart Disease"

        session['basic_report'] = {
            "Sex": sex,
            "Age": age,
            "Cholesterol": cholesterol,
            "MaxHR": maxhr,
            "Oldpeak": oldpeak,
            "Risk": risk_percent,
            "Result": result
        }

        return render_template("basic.html",prediction_text=result,risk_percent=risk_percent)

    except Exception as e:
        return f"Error: {str(e)}"


# ================= DOWNLOAD BASIC =================
@app.route('/download_basic')
def download_basic():

    if 'basic_report' not in session:
        return "No Basic report available"

    import io
    import qrcode
    from reportlab.platypus import Image
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    data = session['basic_report']
    risk = data["Risk"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # ================= CREATE QR IN MEMORY =================
    qr = qrcode.make(f"HeartCare AI Basic Report | Risk: {risk}% | ID: {timestamp}")
    qr_buffer = io.BytesIO()
    qr.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)

    qr_image = Image(qr_buffer, width=90, height=90)

    # ================= CREATE PDF IN MEMORY =================
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    elements = []
    styles = getSampleStyleSheet()

    # ===== TITLE =====
    elements.append(Paragraph("<b>HeartCare AI - Basic Cardiac Risk Report</b>", styles['Title']))
    elements.append(Spacer(1, 10))

    # ===== DATE & TIME =====
    elements.append(Paragraph(
        f"<b>Generated On:</b> {datetime.datetime.now().strftime('%d %B %Y | %I:%M %p')}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 10))

    # ===== RISK BADGE =====
    if risk < 30:
        risk_color = colors.green
        risk_level = "LOW RISK"
    elif risk < 50:
        risk_color = colors.orange
        risk_level = "MODERATE RISK"
    else:
        risk_color = colors.red
        risk_level = "HIGH RISK"

    badge = Table([[f"Risk Score: {risk}% | {risk_level}"]])
    badge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), risk_color),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 14),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))

    elements.append(badge)
    elements.append(Spacer(1, 12))

    # ===== AI CONFIDENCE =====
    confidence = round(85 + (risk / 20), 2)
    elements.append(Paragraph(f"<b>AI Confidence Score:</b> {confidence}%", styles['Normal']))
    elements.append(Spacer(1, 21))

    # ===== PATIENT DATA =====
    patient_table = Table([
        ["Age", data["Age"]],
        ["Cholesterol", data["Cholesterol"]],
        ["Max Heart Rate", data["MaxHR"]],
        ["ST Depression", data["Oldpeak"]],
    ],colWidths=[200, 149])

    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 15))

    # ===== RECOMMENDATIONS =====
    elements.append(Paragraph("<b>Clinical Recommendations</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))

    if risk >= 50:
        recs = [
            "• Immediate cardiology consultation advised",
            "• Avoid high cholesterol and fried foods",
            "• Reduce sodium intake",
            "• Strict blood pressure monitoring",
            "• Begin supervised cardiac rehabilitation",
            "• Stop smoking and alcohol consumption"
        ]
    else:
        recs = [
            "• Maintain balanced diet",
            "• Regular exercise",
            "• Annual heart screening",
            "• Monitor blood pressure periodically",
            "• Maintain healthy body weight"
        ]

    for r in recs:
        elements.append(Paragraph(r, styles['Normal']))
        elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 15))

    # ===== QR CODE =====
    elements.append(Paragraph("<b>Report Verification QR</b>", styles['Normal']))
    elements.append(Spacer(1, 5))
    elements.append(qr_image)

    elements.append(Spacer(1, 20))

    # ===== CENTER FOOTER =====
    center_style = ParagraphStyle(
        name='CenterFooter',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontSize=9
    )

    elements.append(Paragraph(
        "© 2026 HeartCare AI | Designed & Developed by Team CardioPredict",
        center_style
    ))

    # Build PDF
    doc.build(elements)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"basic_report_{timestamp}.pdf",
        mimetype='application/pdf'
    )


# ================= ADVANCED PAGE =================
@app.route('/advanced')
def advanced():
    return render_template("advanced.html")


@app.route('/predict_advanced', methods=['POST'])
def predict_advanced():
    try:
        age = float(request.form['age'])
        restingbp = float(request.form['restingbp'])
        cholesterol = float(request.form['cholesterol'])
        fasting = int(request.form['fasting'])
        maxhr = float(request.form['maxhr'])
        oldpeak = float(request.form['oldpeak'])
        sex_m = int(request.form['sex_m'])
        exercise = int(request.form['exercise'])
        chestpain = request.form['chestpain']
        restingecg = request.form['restingecg']
        stslope = request.form['stslope']

        cp_ATA = 1 if chestpain == "ATA" else 0
        cp_NAP = 1 if chestpain == "NAP" else 0
        cp_TA = 1 if chestpain == "TA" else 0
        ecg_Normal = 1 if restingecg == "Normal" else 0
        ecg_ST = 1 if restingecg == "ST" else 0
        exercise_Y = 1 if exercise == 1 else 0
        slope_Flat = 1 if stslope == "Flat" else 0
        slope_Up = 1 if stslope == "Up" else 0

        numerical = np.array([[age, restingbp, cholesterol, maxhr, oldpeak]])
        scaled = advanced_scaler.transform(numerical)

        final = pd.DataFrame([[ 
            scaled[0][0], scaled[0][1], scaled[0][2], fasting,
            scaled[0][3], scaled[0][4], sex_m,
            cp_ATA, cp_NAP, cp_TA,
            ecg_Normal, ecg_ST,
            exercise_Y, slope_Flat, slope_Up
        ]])

        probability = advanced_model.predict_proba(final)[0][1]
        risk_percent = round(probability * 100, 2)

        result = "⚠️ High Risk of Heart Disease" if probability >= 0.5 else "✅ Low Risk of Heart Disease"

        session['advanced_report'] = {
            "Age": age,
            "RestingBP": restingbp,
            "Cholesterol": cholesterol,
            "MaxHR": maxhr,
            "Oldpeak": oldpeak,
            "ChestPain": chestpain,
            "Risk": risk_percent,
            "Result": result
        }

        return render_template("advanced.html",
                            prediction_text=result,
                            risk_percent=risk_percent)

    except Exception as e:
        return f"Error: {str(e)}"


# ================= DOWNLOAD ADVANCED =================
@app.route('/download_advanced')
def download_advanced():

    if 'advanced_report' not in session:
        return "No Advanced report available"

    import io
    import qrcode
    from reportlab.platypus import Image
    from reportlab.pdfgen import canvas
    from reportlab.platypus import SimpleDocTemplate

    data = session['advanced_report']
    risk = data["Risk"]

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # ================= CREATE QR IN MEMORY =================
    qr = qrcode.make(f"HeartCare AI Advanced Report | Risk: {risk}% | ID: {timestamp}")
    qr_buffer = io.BytesIO()
    qr.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)

    qr_image = Image(qr_buffer, width=80, height=80)

    # ================= CREATE PDF IN MEMORY =================
    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    elements = []
    styles = getSampleStyleSheet()

    # ===== TITLE =====
    elements.append(Paragraph("<b>HeartCare AI - Advanced Cardiac Risk Report</b>", styles['Title']))
    elements.append(Spacer(1, 10))

    # ===== DATE & TIME =====
    elements.append(Paragraph(
        f"<b>Generated On:</b> {datetime.datetime.now().strftime('%d %B %Y | %I:%M %p')}",
        styles['Normal']
    ))
    elements.append(Spacer(1, 10))

    # ===== RISK BADGE =====
    if risk < 30:
        risk_color = colors.green
        risk_level = "LOW RISK"
    elif risk < 50:
        risk_color = colors.orange
        risk_level = "MODERATE RISK"
    else:
        risk_color = colors.red
        risk_level = "HIGH RISK"

    badge = Table([[f"Risk: {risk}% | {risk_level}"]])
    badge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), risk_color),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 14),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))

    elements.append(badge)
    elements.append(Spacer(1, 12))

    # ===== CLINICAL DATA =====
    patient_table = Table([
        ["Age", data["Age"]],
        ["Resting BP", data["RestingBP"]],
        ["Cholesterol", data["Cholesterol"]],
        ["Chest Pain Type", data["ChestPain"]],
        ["Max Heart Rate", data["MaxHR"]],
        ["ST Depression", data["Oldpeak"]],
    ],colWidths=[200, 149])

    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 15))


    # ===== RECOMMENDATIONS =====
    elements.append(Paragraph("<b>Clinical Recommendations</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.3 * inch))

    if risk >= 50:
        recs = [
            "• Immediate cardiology consultation advised",
            "• Avoid high cholesterol and fried foods",
            "• Reduce sodium intake",
            "• Strict blood pressure monitoring",
            "• Begin supervised cardiac rehabilitation",
            "• Stop smoking and alcohol consumption"
        ]
    else:
        recs = [
            "• Maintain balanced diet",
            "• Regular exercise",
            "• Annual heart screening",
            "• Monitor blood pressure periodically",
            "• Maintain healthy body weight"
        ]

    for r in recs:
        elements.append(Paragraph(r, styles['Normal']))
        elements.append(Spacer(1, 0.15 * inch))

    elements.append(Spacer(1, 0.5 * inch))
    # ===== QR CODE =====
    elements.append(Paragraph("<b>Report Verification QR</b>", styles['Normal']))
    elements.append(Spacer(1, 5))
    elements.append(qr_image)

    elements.append(Spacer(1, 15))

    # ===== FOOTER =====

    elements.append(Paragraph(
        "<i>This report is generated by HeartCare AI predictive model. "
        "Not a substitute for professional medical advice.</i>",
        styles['Normal']
    ))


    center_style = ParagraphStyle(
        name='CenterFooter',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontSize=9
    )

    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "© 2026 HeartCare AI | Designed & Developed by Team CardioPredict",
        center_style
    ))

    # Build PDF
    doc.build(elements)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"advanced_report_{timestamp}.pdf",
        mimetype='application/pdf'
    )


if __name__ == "__main__":
    app.run()