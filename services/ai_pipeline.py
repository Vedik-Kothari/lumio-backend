import os
import base64
from groq import AsyncGroq

class AILogic:
    @classmethod
    def get_client(cls):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return AsyncGroq(api_key=api_key)

    @classmethod
    async def transcribe_audio(cls, audio_path: str):
        client = cls.get_client()
        try:
            with open(audio_path, "rb") as file:
                # Using Groq's Whisper API via verbose_json gets timestamps
                transcription = await client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json"
                )
                
                # Some API returns dictionary directly, some object
                if isinstance(transcription, dict):
                    return transcription.get("segments", [])
                else:
                    return getattr(transcription, "segments", [])
        except Exception as e:
            print(f"Transcription error: {e}")
            return []

    @classmethod
    async def caption_image(cls, image_path: str):
        client = cls.get_client()
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = await client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Describe the key events, subjects, and progression across this 2x2 sequential video frame composite. Focus on any motion or scene changes without conversational filler."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=60
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Captioning error: {e}")
            return ""

    @classmethod
    async def search_and_generate(cls, query: str):
        client = cls.get_client()
        from services.vector_store import VectorStore
        
        # 1. Retrieve using Vector Store
        retrieved_chunks = VectorStore.search(query, limit=5)
        
        if not retrieved_chunks:
            return {"answer": "I couldn't find any relevant moments in the source materials.", "sources": []}
            
        context_text = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_text += f"\n[Chunk {i+1} | Timestamp: {chunk['timestamp']}]\n{chunk['text']}\n"
            
        system_prompt = (
            "You are an intelligent Multimodal RAG assistant. You analyze sections of a video/presentation "
            "comprising transcribed audio and visual scene descriptions.\n"
            "Using ONLY the provided context, accurately answer the user's query.\n"
            "CRITICAL: You MUST reference the exact timestamps when explaining where things happen (e.g., 'At 12:40...'). "
            "CRITICAL: Answer in the same language as the user's query or the video context (e.g., if the user asks in Hindi, output your answer in Hindi).\n"
            "Format the output using markdown if necessary."
        )
        
        user_prompt = f"Context:\n{context_text}\n\nQuery: {query}"
        
        try:
            response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return {
                "answer": response.choices[0].message.content,
                "sources": retrieved_chunks
            }
        except Exception as e:
            print(f"Generation error: {e}")
            return {"answer": f"Error generating answer: {e}", "sources": retrieved_chunks}
