from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=200)
    new_password: str = Field(min_length=6, max_length=200)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    system_prompt: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    use_rag: bool = True
    agent_mode: bool = True
    use_web: bool = False
    file_ids: list[int] = []
    stream: bool | None = None


class AnalyzeRequest(BaseModel):
    file_id: int | None = None
    question: str = ""


class ModelDownloadRequest(BaseModel):
    model_id: str = Field(min_length=1)
    kind: str = "llm"


class ModelUseRequest(BaseModel):
    model_id: str = Field(min_length=1)
    kind: str | None = None
    gguf_n_ctx: int | None = Field(default=None, ge=512)
    gguf_n_gpu_layers: int | None = Field(default=None, ge=-1)


class ImageGenRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative: str = ""
    steps: int | None = None
    width: int | None = None
    height: int | None = None
    guidance: float | None = None
    seed: int | None = None
    model_id: str | None = None
    project_id: int | None = None
    chat_id: int | None = None


class SettingsUpdate(BaseModel):
    model: str | None = None
    vision_model: str | None = None
    embedding_model: str | None = None
    image_model: str | None = None
    vision_mode: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    device: str | None = None
    dtype: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    top_k_rag: int | None = None
    history_messages: int | None = None
    context_budget: int | None = None
    agent_router: bool | None = None
    max_tool_rounds: int | None = None
    stream: bool | None = None
    vision_auto_analyze: bool | None = None
    image_steps: int | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_guidance: float | None = None
    image_negative: str | None = None
    hf_token: str | None = None
