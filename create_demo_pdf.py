from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

policies = [
    ("IT and password policy", "Employees can reset a forgotten password through the IT portal. If the reset does not work, contact IT Support via the service desk."),
    ("Vacation policy", "Full-time employees receive 30 working days of paid vacation per calendar year. Submit requests through the HR portal at least two weeks in advance."),
    ("Remote-work policy", "Employees may work remotely up to two days per week after agreeing the schedule with their manager."),
    ("Expense policy", "Business expenses must be submitted within 30 days with a receipt. Manager approval is required above 200 EUR."),
]

styles = getSampleStyleSheet()
story = [Paragraph("Northstar Company - Employee Policies", styles["Title"]), Spacer(1, .5 * cm)]
for heading, text in policies:
    story += [Paragraph(heading, styles["Heading2"]), Paragraph(text, styles["BodyText"]), Spacer(1, .4 * cm)]
SimpleDocTemplate("company_policies_demo.pdf", pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm).build(story)
