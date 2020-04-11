### Required Libraries ###
from datetime import datetime
from dateutil.relativedelta import relativedelta

### Functionality Helper Functions ###
def parse_float(n): # change to float
    """
    Securely converts a non-numeric value to float.
    """
    try:
        return float(n)
    except ValueError:
        return float("nan")
    

def parse_int(n):
    """
    Securely converts a non-integer value to integer.
    """
    try:
        return int(n)
    except ValueError:
        return float("nan")


def build_validation_result(is_valid, violated_slot, message_content):
    """
    Define a result message structured as Lex response.
    """
    if message_content is None:
        return {"isValid": is_valid, "violatedSlot": violated_slot}

    return {
        "isValid": is_valid,
        "violatedSlot": violated_slot,
        "message": {"contentType": "PlainText", "content": message_content},
    }

    
def validate_data(age, investment_amount, intent_request):
    """
    Validates the data provided by the user.
    """

    # Validate that the user is at least 18 years old in age
    if age is not None:
        age = parse_int(age) 
        # "parse_int" defined before to return INT

        if age < 18:
            return build_validation_result(
                False,
                "age",
                "Could you please contact your parents to open a custodial account?"
                " It can be transferred under your name after your 18th birthday. Thank you :P"
            )
        
        elif age < 0:
            return build_validation_result(
                False,
                "age",
                "Could you please provide an age greater than zero?"
                " Recommendations for unborn babies"
                " will be available in the future. Your patience is greatly appreciated."
                )

    # Validate the investment amount, it should be >=5000
    if investment_amount is not None:
        investment_amount = parse_float(
            investment_amount
        )  # Since parameters are strings, it's important to cast values. Function "parse_float" is defined at the beginning
        if investment_amount < 5000:
            return build_validation_result(
                False,
                "investmentAmount",
                "Could you provide an investment amount equal to or greater than $5,000, please?"
                " Investment options for smaller amounts will be available in the future :)"
            )

    # A True results is returned if age or amount are valid
    return build_validation_result(True, None, None)


### Dialog Actions Helper Functions ###
def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    """
    Defines an elicit slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "ElicitSlot",
            "intentName": intent_name,
            "slots": slots,
            "slotToElicit": slot_to_elicit,
            "message": message,
        },
    }


def delegate(session_attributes, slots):
    """
    Defines a delegate slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": "Delegate", "slots": slots},
    }


def close(session_attributes, fulfillment_state, message):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillment_state,
            "message": message,
        },
    }

    return response


### Intents Handlers ###
def recommend_portfolio(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    first_name = get_slots(intent_request)["firstName"]
    age = get_slots(intent_request)["age"]
    investment_amount = get_slots(intent_request)["investmentAmount"]
    risk_level = get_slots(intent_request)["riskLevel"]
    source = intent_request["invocationSource"]

    if source == "DialogCodeHook":
        
        ### YOUR DATA VALIDATION CODE STARTS HERE ###

        # Perform basic validation on the supplied input slots.
        # Use the elicitSlot dialog action to re-prompt
        # for the first violation detected.

       # Gets all the slots
        slots = get_slots(intent_request)
        
        # Validates user's input using the validate_data function
        validation_result = validate_data(age, investment_amount, intent_request)
        
         # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        if not validation_result["isValid"]:
            slots[validation_result["violatedSlot"]] = None  # Cleans invalid slot
            
# Returns an elicitSlot dialog to request new data for the invalid slot
            return elicit_slot(
                intent_request["sessionAttributes"],
                intent_request["currentIntent"]["name"],
                slots,
                validation_result["violatedSlot"],
                validation_result["message"],
            )

        ### YOUR DATA VALIDATION CODE ENDS HERE ###

        # Fetch current session attibutes
        output_session_attributes = intent_request["sessionAttributes"]

        return delegate(output_session_attributes, get_slots(intent_request))
    
    ### YOUR FINAL INVESTMENT RECOMMENDATION CODE STARTS HERE ###
    
    # define investment portfolio recommendations
    #def recommended_portfolios(risk):
    """
    Responses based on selected risk levels.
    """
    risk = {
        "None": "100% bonds (AGG), 0% equities (SPY)",
        "Very Low": "80% bonds (AGG), 20% equities (SPY)",
        "Low": "60% bonds (AGG), 40% equities (SPY)",
        "Medium": "40% bonds (AGG), 60% equities (SPY)",
        "High": "20% bonds (AGG), 80% equities (SPY)",
        "Very High": "0% bonds (AGG), 100% equities (SPY)"
    }
    #   return risk[risk_level]

# Get the initial investment recommendation
    initial_recommendation = risk[risk_level]

    # Return a message with the initial recommendation based on the risk level.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
           {
            "contentType": "PlainText",
            "content": """{} thank you for your information;
            based on your preferred risk, an investment portfolio is customized for you: {}
            """.format(
                first_name, initial_recommendation
            ),
        },
    )
    ### YOUR FINAL INVESTMENT RECOMMENDATION CODE ENDS HERE ###
    
### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "RecommendPortfolio":
        return recommend_portfolio(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
