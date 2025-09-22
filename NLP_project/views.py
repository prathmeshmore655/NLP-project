from django.http import JsonResponse
from .second_model import ChatbotService
from rest_framework.decorators import api_view


@api_view(['GET'])
def root_view(request):
    return JsonResponse({"message": "Welcome to the NLP Chatbot API!"})


@api_view(['POST'])
def chatbot_view(request):
    user_message = request.data.get("message", "")
    if not user_message:
        return JsonResponse({"error": "No message provided."}, status=400)

    chatbot_service = ChatbotService()
    response = chatbot_service.process_chat_message(user_message, user_id="test_user")

    return JsonResponse({"response": response})

    
    

    # Here you would integrate your NLP model to generate a response
    # For demonstration, we'll just echo the user's message
    

