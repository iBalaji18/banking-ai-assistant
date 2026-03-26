import json
import random

# BFSI Dataset Categories
# 1. Loan eligibility and application status
# 2. EMI details and schedules
# 3. Interest rate information
# 4. Payment and transaction queries
# 5. Basic account and customer support

def generate_samples():
    dataset = []

    # 1. Loan eligibility
    loan_types = ["personal loan", "home loan", "car loan", "education loan", "business loan"]
    loan_statuses = ["pending", "approved", "rejected", "under review"]
    
    for _ in range(35):
        l_type = random.choice(loan_types)
        amount = random.randint(1, 50) * 100000
        dataset.append({
            "instruction": f"Answer the customer's query regarding their {l_type} eligibility.",
            "input": f"Am I eligible for a {l_type} of {amount} INR?",
            "output": f"Eligibility for a {l_type} of {amount} INR depends on your credit score, current income, and existing liabilities. Please provide your net monthly salary and employment details so we can assist you better."
        })
        
    for _ in range(15):
        l_type = random.choice(loan_types)
        dataset.append({
            "instruction": "Provide the application status for the loan.",
            "input": f"What is the status of my {l_type} application?",
            "output": "To check the exact status of your application, please provide your 10-digit application reference number. Standard processing times are 3-5 business days."
        })

    # 2. EMI Details
    for _ in range(25):
        dataset.append({
            "instruction": "Answer the customer's query about EMI.",
            "input": "How can I change my EMI debit date?",
            "output": "You can change your EMI mandate date by logging into the net banking portal under 'Loans' -> 'Update Mandate', or by visiting your nearest branch. Please note that date changes take effect from the next billing cycle."
        })
    for _ in range(15):
        dataset.append({
            "instruction": "Respond to the EMI bounce query.",
            "input": "My EMI bounced this month. What are the charges?",
            "output": "If an EMI bounces due to insufficient funds, a standard bounce charge of INR 500 plus applicable taxes is levied. We recommend maintaining sufficient balance 24 hours prior to the EMI date. Please make the payment manually via the app immediately to avoid late payment interest."
        })

    # 3. Interest rates
    for _ in range(30):
        l_type = random.choice(loan_types)
        dataset.append({
            "instruction": f"Provide information about {l_type} interest rates.",
            "input": f"What are the current interest rates for a {l_type}?",
            "output": f"Our {l_type} interest rates start from 8.5% p.a., subject to your credit profile, loan amount, and tenure. For exact customized rates, please visit our website or speak to a relationship manager."
        })

    # 4. Payment and Transactions
    for _ in range(20):
        dataset.append({
            "instruction": "Assist with a transaction query.",
            "input": "Money was deducted from my account but the transaction failed.",
            "output": "We understand your concern. Failed transactions where money is debited are usually automatically reversed to your account within 3 to 5 business days. If it has been longer, please raise a dispute via the mobile app."
        })
        
    # 5. Basic Account Support
    for _ in range(25):
        dataset.append({
            "instruction": "Help the customer with account support.",
            "input": "How do I update my registered mobile number and email ID?",
            "output": "To update your mobile number or email, you can use an ATM with your debit card and PIN, or log into Internet Banking using your existing credentials. For security reasons, you may also need to complete an OTP verification."
        })
        
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    
    data = generate_samples()
    print(f"Generated {len(data)} samples.")
    
    with open("data/bfsi_dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("Saved to data/bfsi_dataset.json")
