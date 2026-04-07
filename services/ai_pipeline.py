import os
import base64
import asyncio
from groq import AsyncGroq

class AILogic:
    VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    GENERATION_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.1-8b-instant")

    @classmethod
    def get_client(cls):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return AsyncGroq(api_key=api_key)

    @classmethod
    async def _create_chat_completion_with_retry(cls, **kwargs):
        client = cls.get_client()
        last_error = None

        for attempt in range(3):
            try:
                return await client.chat.completions.create(**kwargs)
            except Exception as e:
                last_error = e
                if attempt == 2:
                    raise
                await asyncio.sleep(1.5 * (attempt + 1))

        raise last_error

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
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = await cls._create_chat_completion_with_retry(
                model=cls.VISION_MODEL,
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
            return "Vision caption unavailable for this moment."

    @classmethod
    async def search_and_generate(cls, query: str, video_id: str | None = None):
        from services.vector_store import VectorStore
        
        # 1. Retrieve using Vector Store
        retrieved_chunks = VectorStore.search(query, limit=5, video_id=video_id)
        
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
            response = await cls._create_chat_completion_with_retry(
                model=cls.GENERATION_MODEL,
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
            fallback_lines = [
                "I couldn't reach the language model reliably, but I did retrieve the most relevant indexed moments.",
                "",
            ]
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                fallback_lines.append(
                    f"{idx}. {chunk['timestamp']}: {chunk['text'].strip()[:280]}"
                )

            return {"answer": "\n".join(fallback_lines), "sources": retrieved_chunks}

    @classmethod
    def _build_workbench_prompt(
        cls,
        *,
        mode: str,
        query: str | None,
        output_format: str | None,
        scope: str,
    ) -> tuple[str, str]:
        mode_instructions = {
            "moment_map": "Create a timestamped moment map with turning points, topic shifts, important visuals, decisions, and notable evidence anchors.",
            "save_time": "Tell the user whether the video is worth watching, give a short version, and explicitly call out what can be skipped.",
            "output_generator": f"Transform the content into a high-quality {output_format or 'structured notes'} with practical formatting.",
            "ask_for_me": "Act like a one-tap agent. Complete the user's requested task directly using the video context and surface exact timestamps.",
            "cross_video_memory": "Compare and synthesize themes across multiple videos. Highlight agreements, contradictions, and recurring insights.",
            "learning_mode": "Turn the material into a learning pack with key concepts, quiz questions, flashcards, and a short study plan.",
            "decision_mode": "Separate facts, claims, speculation, actions, risks, and missing evidence. Be precise and evidence-locked.",
            "resource_extractor": "Extract all practical resources, tools, products, people, links, books, deadlines, metrics, and references mentioned.",
            "what_changed": "Identify what changed over time, what was updated, and where the important transitions happened.",
            "presentation_mode": "Rewrite the video into a clean executive briefing with goals, sections, highlights, and takeaway actions.",
        }
        system_prompt = (
            "You are a multimodal video intelligence workbench.\n"
            "Use only the provided context.\n"
            "Every substantive claim must reference timestamps.\n"
            "Prefer concise structure, markdown headings, and bullets when helpful.\n"
            "If context is weak or missing, say so plainly.\n"
        )
        user_prompt = (
            f"Mode: {mode}\n"
            f"Scope: {scope}\n"
            f"Task: {mode_instructions.get(mode, 'Analyze the video context helpfully and precisely.')}\n"
            f"User query: {query or 'None provided'}\n"
        )
        return system_prompt, user_prompt

    @classmethod
    async def run_workbench(
        cls,
        *,
        mode: str,
        video_id: str | None = None,
        query: str | None = None,
        output_format: str | None = None,
        scope: str = "current_video",
    ):
        from services.vector_store import VectorStore

        if mode == "cross_video_memory":
            source_chunks = VectorStore.search(query or "main themes and differences", limit=12, video_id=None)
        elif query:
            source_chunks = VectorStore.search(query, limit=10, video_id=video_id if scope != "all_videos" else None)
        else:
            source_chunks = VectorStore.get_chunks(video_id=video_id if scope != "all_videos" else None, limit=120)

        if not source_chunks:
            return {
                "title": "No context available",
                "content": "I couldn't find enough indexed material for that workbench task yet.",
                "sources": [],
                "mode": mode,
            }

        context_blocks = []
        for index, chunk in enumerate(source_chunks, start=1):
            context_blocks.append(
                f"[Chunk {index} | Video: {chunk.get('video_id', 'unknown')} | Timestamp: {chunk.get('timestamp', 'n/a')}]\n"
                f"{chunk.get('text', '')}"
            )

        system_prompt, user_prompt = cls._build_workbench_prompt(
            mode=mode,
            query=query,
            output_format=output_format,
            scope=scope,
        )
        full_prompt = user_prompt + "\nContext:\n" + "\n\n".join(context_blocks)

        try:
            response = await cls._create_chat_completion_with_retry(
                model=cls.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"Workbench generation error: {e}")
            fallback_lines = [
                "The workbench model was unavailable, so here are the most relevant grounded chunks:",
                "",
            ]
            for index, chunk in enumerate(source_chunks[:8], start=1):
                fallback_lines.append(
                    f"{index}. [{chunk.get('timestamp', 'n/a')}] {chunk.get('text', '').strip()[:320]}"
                )
            content = "\n".join(fallback_lines)

        titles = {
            "moment_map": "Moment Map",
            "save_time": "Save Me Time",
            "output_generator": output_format or "Output Generator",
            "ask_for_me": "Ask For Me",
            "cross_video_memory": "Cross-Video Memory",
            "learning_mode": "Learning Mode",
            "decision_mode": "Decision Mode",
            "resource_extractor": "Resource Extractor",
            "what_changed": "What Changed",
            "presentation_mode": "Presentation Mode",
        }
        return {
            "title": titles.get(mode, "Workbench"),
            "content": content,
            "sources": source_chunks[:10],
            "mode": mode,
        }
