"""
FinBERT Analyzer Module
Processes invoices, emails, and financial documents for spend analysis
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import json
from typing import Dict, List, Tuple
from datetime import datetime

class FinBERTAnalyzer:
    def __init__(self):
        """Initialize FinBERT model for financial text analysis"""
        print("Loading FinBERT model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model.to(self.device)
        self.model.eval()
        
        # Load categories configuration
        self.load_categories()
        
        # Sentiment labels
        self.sentiment_labels = ['positive', 'negative', 'neutral']
        
        print(f"Model loaded on {self.device}")
    
    def load_categories(self):
        """Load spend categories and keywords from config"""
        try:
            with open('config/categories.json', 'r') as f:
                config = json.load(f)
                self.categories = config['categories']
                self.keywords = config['keywords']
        except:
            # Default categories if config not found
            self.categories = [
                "Cloud Services", "Software", "Marketing", 
                "Office", "Travel", "Other"
            ]
            self.keywords = {
                "Cloud Services": ["aws", "azure", "google cloud", "hosting"],
                "Software": ["license", "subscription", "saas", "software"]
            }
    
    def extract_entities(self, text: str) -> Dict:
        """Extract vendor, amount, and dates from text"""
        entities = {
            'vendor': None,
            'amount': None,
            'date': None,
            'invoice_number': None
        }
        
        # Extract vendor (company names - simple pattern)
        vendor_pattern = r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b'
        vendors = re.findall(vendor_pattern, text[:200])  # Check first 200 chars
        if vendors:
            entities['vendor'] = vendors[0]
        
        # Extract amounts
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?)\b'
        amounts = re.findall(amount_pattern, text)
        if amounts:
            # Clean and get largest amount (likely total)
            clean_amounts = []
            for amt in amounts:
                clean = re.sub(r'[^\d.]', '', amt)
                if clean:
                    clean_amounts.append(float(clean))
            if clean_amounts:
                entities['amount'] = max(clean_amounts)
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            entities['date'] = dates[0]
        
        # Extract invoice number
        invoice_pattern = r'(?:Invoice|INV|Inv)[\s#-]*(\w+)'
        invoice_matches = re.findall(invoice_pattern, text, re.IGNORECASE)
        if invoice_matches:
            entities['invoice_number'] = invoice_matches[0]
        
        return entities
    
    def categorize_spend(self, text: str) -> str:
        """Categorize spending based on text content"""
        text_lower = text.lower()
        
        # Check keywords for each category
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        # Use vendor name for categorization if found
        if 'aws' in text_lower or 'amazon web' in text_lower:
            return 'Cloud Services'
        elif any(word in text_lower for word in ['microsoft', 'adobe', 'slack', 'zoom']):
            return 'Software'
        elif any(word in text_lower for word in ['hotel', 'flight', 'uber', 'taxi']):
            return 'Travel'
        
        return 'Other'
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of financial text"""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get sentiment scores
        scores = predictions[0].cpu().numpy()
        sentiment_scores = {
            self.sentiment_labels[i]: float(scores[i]) 
            for i in range(len(self.sentiment_labels))
        }
        
        # Determine primary sentiment
        primary_sentiment = self.sentiment_labels[scores.argmax()]
        
        # Calculate risk score based on negative sentiment
        risk_score = scores[1] * 100  # Negative sentiment as risk
        
        return {
            'sentiment': primary_sentiment,
            'scores': sentiment_scores,
            'risk_score': risk_score
        }
    
    def detect_anomalies(self, amount: float, vendor: str, historical_data: pd.DataFrame = None) -> Dict:
        """Detect spending anomalies"""
        anomaly_result = {
            'is_anomaly': False,
            'confidence': 0.0,
            'reason': None
        }
        
        if historical_data is not None and not historical_data.empty:
            # Filter historical data for this vendor
            vendor_history = historical_data[historical_data['vendor'] == vendor]
            
            if not vendor_history.empty:
                mean_amount = vendor_history['amount'].mean()
                std_amount = vendor_history['amount'].std()
                
                # Check if amount is outside 2 standard deviations
                if std_amount > 0:
                    z_score = abs(amount - mean_amount) / std_amount
                    if z_score > 2:
                        anomaly_result['is_anomaly'] = True
                        anomaly_result['confidence'] = min(z_score / 3, 1.0)  # Normalize to 0-1
                        
                        if amount > mean_amount:
                            anomaly_result['reason'] = f"Amount ${amount:.2f} is {z_score:.1f} standard deviations above average ${mean_amount:.2f}"
                        else:
                            anomaly_result['reason'] = f"Amount ${amount:.2f} is {z_score:.1f} standard deviations below average ${mean_amount:.2f}"
                
                # Check for sudden increases
                if amount > mean_amount * 1.3:  # 30% increase
                    anomaly_result['is_anomaly'] = True
                    anomaly_result['confidence'] = 0.7
                    increase_pct = ((amount - mean_amount) / mean_amount) * 100
                    anomaly_result['reason'] = f"Sudden {increase_pct:.1f}% increase from average"
        
        return anomaly_result
    
    def process_document(self, text: str, historical_data: pd.DataFrame = None) -> Dict:
        """Main processing function for a single document"""
        # Extract entities
        entities = self.extract_entities(text)
        
        # Categorize spending
        category = self.categorize_spend(text)
        
        # Analyze sentiment
        sentiment_analysis = self.analyze_sentiment(text)
        
        # Detect anomalies if we have amount and vendor
        anomaly_detection = {'is_anomaly': False}
        if entities['amount'] and entities['vendor'] and historical_data is not None:
            anomaly_detection = self.detect_anomalies(
                entities['amount'], 
                entities['vendor'], 
                historical_data
            )
        
        # Compile results
        result = {
            'timestamp': datetime.now().isoformat(),
            'vendor': entities['vendor'],
            'amount': entities['amount'],
            'date': entities['date'],
            'invoice_number': entities['invoice_number'],
            'category': category,
            'sentiment': sentiment_analysis['sentiment'],
            'sentiment_scores': sentiment_analysis['scores'],
            'risk_score': sentiment_analysis['risk_score'],
            'is_anomaly': anomaly_detection['is_anomaly'],
            'anomaly_confidence': anomaly_detection.get('confidence', 0),
            'anomaly_reason': anomaly_detection.get('reason', None)
        }
        
        return result
    
    def batch_process(self, documents: List[str], batch_size: int = 32) -> pd.DataFrame:
        """Process multiple documents in batches"""
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            for doc in batch:
                try:
                    result = self.process_document(doc)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing document: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights from processed data"""
        insights = {
            'total_spend': df['amount'].sum() if 'amount' in df else 0,
            'vendor_count': df['vendor'].nunique() if 'vendor' in df else 0,
            'top_categories': {},
            'risk_vendors': [],
            'anomalies': [],
            'recommendations': []
        }
        
        # Top spending categories
        if 'category' in df and 'amount' in df:
            category_spend = df.groupby('category')['amount'].sum().sort_values(ascending=False)
            insights['top_categories'] = category_spend.head(5).to_dict()
        
        # High risk vendors
        if 'risk_score' in df:
            high_risk = df[df['risk_score'] > 50]
            if not high_risk.empty:
                insights['risk_vendors'] = high_risk[['vendor', 'risk_score']].to_dict('records')
        
        # Anomalies
        if 'is_anomaly' in df:
            anomalies = df[df['is_anomaly'] == True]
            if not anomalies.empty:
                insights['anomalies'] = anomalies[['vendor', 'amount', 'anomaly_reason']].to_dict('records')
        
        # Generate recommendations
        if insights['anomalies']:
            insights['recommendations'].append("Review flagged anomalous transactions immediately")
        
        if insights['risk_vendors']:
            insights['recommendations'].append("Schedule reviews with high-risk vendors")
        
        if 'amount' in df:
            top_vendors = df.groupby('vendor')['amount'].sum().sort_values(ascending=False).head(3)
            if top_vendors.sum() / df['amount'].sum() > 0.6:
                insights['recommendations'].append("Consider diversifying vendor portfolio - high concentration risk")
        
        return insights


# Standalone usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = FinBERTAnalyzer()
    
    # Sample invoice text
    sample_text = """
    INVOICE #INV-2024-1234
    Date: November 15, 2024
    
    Bill To: Your Company Inc.
    From: Amazon Web Services
    
    Cloud Computing Services - November 2024
    EC2 Instances: $3,500.00
    S3 Storage: $1,200.00
    RDS Database: $2,800.00
    
    Total Due: $7,500.00
    
    Payment Terms: Net 30
    Thank you for your business. Costs have increased this month due to higher usage.
    """
    
    # Process the document
    result = analyzer.process_document(sample_text)
    
    # Print results
    print("\n=== Analysis Results ===")
    print(f"Vendor: {result['vendor']}")
    print(f"Amount: ${result['amount']:,.2f}")
    print(f"Category: {result['category']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Risk Score: {result['risk_score']:.1f}%")
    print(f"Invoice #: {result['invoice_number']}")
    
    if result['is_anomaly']:
        print(f"\n⚠️ ANOMALY DETECTED: {result['anomaly_reason']}")
    
    print("\n=== Sentiment Breakdown ===")
    for sent, score in result['sentiment_scores'].items():
        print(f"{sent}: {score:.3f}")