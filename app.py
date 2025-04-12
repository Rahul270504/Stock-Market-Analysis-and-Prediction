import yfinance as yf
import json
import pandas as pd
import logging
from google import genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/stock_analysis_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_analyzer")

try:
    API_KEY = "AIzaSyCS2v_XwtNWd_K6aCe6Minex-v1ovsA2mw"
    client = genai.Client(api_key=API_KEY)
    logger.info("Gemini API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API client: {str(e)}")
    logger.error(traceback.format_exc())

def calculate_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        loss = loss.replace(0, 0.001) 
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series([50] * len(prices), index=prices.index) 

def get_enhanced_data(company_name):
    logger.info(f"Fetching enhanced data for {company_name}")
    try:
        stock = yf.Ticker(company_name)
        hist = stock.history(period="6mo")
        
        if hist.empty:
            logger.warning(f"No historical data found for {company_name}")
            return pd.DataFrame(), {}
        
        logger.info(f"Successfully fetched {len(hist)} data points for {company_name}")
        
        try:
            company_info = stock.info
            logger.info(f"Successfully fetched company info for {company_name}")
        except Exception as e:
            logger.warning(f"Failed to fetch company info for {company_name}: {str(e)}")
            company_info = {"shortName": company_name, "sector": "Unknown"}
        
        if len(hist) > 0:
            hist['MA5'] = hist['Close'].rolling(window=5).mean()
            hist['MA10'] = hist['Close'].rolling(window=10).mean()
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            
            hist['RSI'] = calculate_rsi(hist['Close'])
            
            hist['MA20_std'] = hist['Close'].rolling(window=20).std()
            hist['upper_band'] = hist['MA20'] + (hist['MA20_std'] * 2)
            hist['lower_band'] = hist['MA20'] - (hist['MA20_std'] * 2)
            
            hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
            hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
            hist['MACD'] = hist['EMA12'] - hist['EMA26']
            hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            hist['MACD_histogram'] = hist['MACD'] - hist['Signal']
            
            logger.info(f"Successfully calculated technical indicators for {company_name}")
        
        return hist, company_info
    except Exception as e:
        logger.error(f"Error in get_enhanced_data for {company_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), {"shortName": company_name, "sector": "Unknown"}

def analyze_stock(company_name):
    logger.info(f"Analyzing stock for {company_name}")
    try:
        hist, company_info = get_enhanced_data(company_name)
        
        if hist.empty:
            logger.warning(f"No data to analyze for {company_name}")
            return json.dumps({"error": f"No data found for {company_name}"})
        
        start_date = hist.index[0].strftime('%Y-%m-%d')
        end_date = hist.index[-1].strftime('%Y-%m-%d')
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        
        logger.info(f"Preparing analysis prompt for {company_name}")
        
        company_data = hist.to_string()
        
        prompt_for_analysis = f"""
        You are a good analyst. You will be given a company name and its stock information for a month.
        Your task is to analyze the data and provide insights on the company's performance.
        Here is the data: {company_data}
        
        Format your response as a JSON object with the following structure:
        {{
          "company_info": {{
            "name": "{company_info.get('shortName', company_name)}",
            "ticker": "{company_name}",
            "sector": "{company_info.get('sector', 'Unknown')}"
          }},
          "time_period_analyzed": "{start_date} to {end_date}",
          "price_metrics": {{
            "starting_price": {start_price:.2f},
            "ending_price": {end_price:.2f},
            "percent_change": {percent_change:.2f},
            "high": {hist['High'].max():.2f},
            "low": {hist['Low'].min():.2f},
            "average": {hist['Close'].mean():.2f}
          }},
          "volume_metrics": {{
            "average_volume": {int(hist['Volume'].mean())},
            "highest_volume_day": "{hist['Volume'].idxmax().strftime('%Y-%m-%d')}",
            "lowest_volume_day": "{hist['Volume'].idxmin().strftime('%Y-%m-%d')}",
            "volume_trend": "analyze_this_manually"
          }},
          "technical_indicators": {{
            "moving_averages": {{
              "ma_5": {hist['MA5'].iloc[-1]:.2f},
              "ma_10": {hist['MA10'].iloc[-1]:.2f},
              "ma_20": {hist['MA20'].iloc[-1]:.2f},
              "ma_trend": "analyze_this_manually"
            }},
            "rsi": {{
              "current": {hist['RSI'].iloc[-1]:.2f},
              "interpretation": "analyze_this_manually"
            }},
            "macd": {{
              "current": {hist['MACD'].iloc[-1]:.2f},
              "signal": {hist['Signal'].iloc[-1]:.2f},
              "histogram": {hist['MACD_histogram'].iloc[-1]:.2f},
              "trend": "analyze_this_manually"
            }},
            "bollinger_bands": {{
              "upper": {hist['upper_band'].iloc[-1]:.2f},
              "middle": {hist['MA20'].iloc[-1]:.2f},
              "lower": {hist['lower_band'].iloc[-1]:.2f},
              "position": "analyze_this_manually"
            }}
          }},
          "volatility_analysis": {{
            "standard_deviation": {hist['Close'].std():.2f},
            "volatility_description": "analyze_this_manually"
          }},
          "pattern_recognition": {{
            "identified_patterns": ["analyze_this_manually"],
            "support_levels": [0.0],
            "resistance_levels": [0.0]
          }},
          "risk_assessment": {{
            "level": "analyze_this_manually",
            "factors": ["analyze_this_manually"]
          }},
          "summary": {{
            "overall_trend": "analyze_this_manually",
            "key_insights": [
              "analyze_this_manually"
            ],
            "recommendation": "analyze_this_manually",
            "confidence_level": "analyze_this_manually",
            "outlook": "analyze_this_manually"
          }}
        }}
        
        Fill in the "analyze_this_manually" placeholders with your analysis. Provide actual values for support_levels and resistance_levels based on your analysis.
        """
        
        logger.info(f"Analysis prompt prepared for {company_name}")
        return prompt_for_analysis
    except Exception as e:
        logger.error(f"Error in analyze_stock for {company_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return json.dumps({"error": f"Failed to analyze stock for {company_name}: {str(e)}"})

def llm_call(analysis_prompt):
    logger.info("Calling LLM for analysis")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[analysis_prompt],
        )
        logger.info("LLM response received successfully")
        output = response.text.strip()[7: -3]
        
        try:
            json_output = json.loads(output)
            logger.info("LLM response parsed successfully as JSON")
            return json_output
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"Raw response: {output}")
            return {"error": "Failed to parse LLM response as JSON", "raw_response": output}
    except Exception as e:
        logger.error(f"Error in LLM call: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Failed to get LLM analysis: {str(e)}"}

def get_analysis(company_name):
    logger.info(f"Starting analysis for {company_name}")
    try:
        analysis_prompt = analyze_stock(company_name)
        if isinstance(analysis_prompt, str) and analysis_prompt.startswith('{"error"'):
            return json.loads(analysis_prompt)
        
        logger.info(f"Sending analysis prompt to LLM for {company_name}")
        llm_output = llm_call(analysis_prompt)
        logger.info(f"Analysis completed for {company_name}")
        return llm_output
    except Exception as e:
        logger.error(f"Error in get_analysis for {company_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Failed to complete analysis for {company_name}: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data in request")
            return jsonify({"error": "No data provided"}), 400
        
        company_name = data.get('company_name')
        if not company_name:
            logger.error("No company name provided in request")
            return jsonify({"error": "No company name provided"}), 400
        
        logger.info(f"Received analysis request for {company_name}")
        
        analysis_result = get_analysis(company_name)
        
        if "error" in analysis_result:
            logger.error(f"Analysis failed for {company_name}: {analysis_result['error']}")
            return jsonify(analysis_result), 500
        
        logger.info(f"Successfully analyzed {company_name}")
        return jsonify(analysis_result)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

if __name__ == "__main__":
    port = 5000
    debug = True
    
    logger.info(f"Starting Stock Analysis API on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)