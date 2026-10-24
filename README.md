# Gpshap

### Your local AI workspace for private chats, projects, documents, and image creation

Gpshap is a Windows-first application for running language models on your own computer. It combines a browser-based chat interface with local model management, project-aware document search, image generation, and optional web search.

> **Local by default.** Conversation history, project resources, and generated images are stored on your computer. Gpshap does not upload or email chat transcripts.

<p align="center">
  <strong>Chat locally · Organize by project · Search your files · Create images</strong>
</p>

---

## ✨ What you can do

| Capability | Description |
| --- | --- |
| **Local AI chat** | Chat with a model running on your computer using a streaming interface. |
| **Transformers and GGUF models** | Use Hugging Face/Transformers models or local `.gguf` models through llama.cpp. |
| **Chats and projects** | Keep general chats outside projects, or create multiple chats inside a project. Existing project conversations are preserved during database migration. |
| **Project documents (RAG)** | Add documents to a project and ask questions grounded in their extracted text. Project chats share that project's resources. |
| **Image understanding** | Upload images to a project and use the configured vision model to analyze them. |
| **Image Studio** | Generate images locally with a Diffusers text-to-image model. The first generation downloads the model components and reports progress. |
| **Model Hub** | Browse, download, select, cancel, and remove supported models from the local UI. |
| **Optional agent tools** | Use available local tools such as document search and calculations while chatting. |
| **Optional web search** | Enable search for a specific message when you need current information. |
| **User accounts** | Register and sign in to keep each user's local chats and projects separate. |
| **Local resource monitoring** | View CPU, memory, GPU, and disk information when the hardware and drivers expose those metrics. |
| **Local export and feedback** | Export chats as JSON. The feedback button opens a pre-addressed email draft for you to review and send. |

The application is intended for local, single-computer use. It is not a hosted service and does not provide encrypted database storage or remote account synchronization.

## 🖥️ Requirements

### Required

- **Windows 10 or Windows 11, 64-bit**
- An internet connection for the initial setup and for downloading models
- Enough free disk space for the Python environment, dependencies, and selected model files. Model sizes vary widely; check the Models page before downloading larger models.
- A supported browser, such as Microsoft Edge, Chrome, or Firefox

The installer downloads its own **64-bit Python 3.12.10** into the project directory and creates `.venv312`. You do not need to install Python separately.

### Hardware guidance

Model performance and memory requirements depend on the selected model, its quantization, context length, and enabled features. A GPU is optional; CPU inference is supported but can be slower. NVIDIA users need a compatible NVIDIA driver for CUDA acceleration. If the installer cannot use the CUDA wheels, it falls back to CPU PyTorch wheels.

For a smoother experience, use a modern multi-core CPU, **16 GB or more of system RAM**, and an NVIDIA GPU with as much VRAM as your chosen models require. These are recommendations, not fixed minimums; smaller models can run on less capable hardware, while larger models may need substantially more memory and disk space.

> **Memory use:** Gpshap unloads the text-generation model before starting image generation, and unloads the image pipeline before starting chat generation. The vision and embedding features can still have their own resource needs.

## 🚀 Installation

### 1. Download the project

Download the latest source from GitHub using **Code → Download ZIP**, then extract it to a folder you can write to, for example:

```text
C:\Users\<your-name>\Gpshap
```

Alternatively, use Git:

```powershell
git clone https://github.com/nkyzai/gpshap.git
cd gpshap
```

### 2. Run the installer

Open the extracted folder and run **`install.bat`**. It installs the project-local Python 3.12 environment, PyTorch, and the application dependencies. By default, it also downloads the default text model and embedding model.

Installer options:

| Command | Effect |
| --- | --- |
| `.\install.bat` | Set up dependencies and download the default text and embedding models. |
| `.\install.bat --no-models` | Set up dependencies without bootstrapping models. Download models later in the Models page. |
| `.\install.bat --vision` | Also download the optional vision model during setup. |
| `.\install.bat --imagegen` | Also download the optional image-generation model during setup. |
| `.\install.bat --all` | Also download the optional vision and image-generation models. |
| `.\install.bat --cpu` | Install CPU-only PyTorch even if NVIDIA hardware is detected. |

Options can be combined. For example, `install.bat --no-models --cpu` installs the CPU environment and defers model downloads.

The installer requires internet access and may take time, especially when downloading PyTorch or model files. If installation fails, read the error shown in the installer window, check your connection and available disk space, and run `install.bat` again after addressing the issue.

### 3. Start Gpshap

After installation, run:

```text
run.bat
```

The application opens in your browser at **[http://127.0.0.1:8787](http://127.0.0.1:8787)**. Keep the command window open while using Gpshap; closing it stops the local server.

### 4. Create an account

Register a local account from the sign-in screen, then sign in. Accounts and their associated chats are stored in the local application database; there is no cloud account service.

## 📖 How to use Gpshap

### Chat and organize your work

1. Click **+ Chat** to start a general conversation. General chats do not need a project.
2. Click **+ Project** to create a project, then use **+ New chat** under that project to create project-specific conversations.
3. Select a project name to show its chats. General chats remain visible independently.
4. Use the chat's **⋯** menu to rename, duplicate, export, clear, or delete it.
5. Use the project **Resources** button to inspect its files. Project resources are available to chats within that project.

### Choose a language model

1. Open **Models**.
2. Browse available models or inspect the local model inventory.
3. Download a Transformers model or choose a local `.gguf` file.
4. Select **Use** for the model you want, then return to Chat.

GGUF models are run through llama.cpp. Model compatibility and performance vary by architecture, quantization, available RAM/VRAM, and llama.cpp wheel support.

### Ask questions about project files

1. Open a chat inside a project.
2. Click the attachment button or **Upload to selected project**.
3. Enable **RAG** in the chat controls when you want the model to retrieve relevant passages from that project's indexed documents.
4. Ask a question and inspect the sources attached to the response.

Supported extraction depends on file type and installed libraries. Image analysis also requires the configured vision model to be available.

### Generate an image

1. Open **Studio** and enter a prompt.
2. Choose image settings and click **Generate**.
3. On first use, Gpshap downloads the configured image model and required pipeline components; the page displays download and generation status.
4. Generated images are saved locally and appear in the gallery. When a chat is active, the result is also attached to that chat.

The first image generation can take a while. Image generation may require significant memory and can be slow on CPU-only systems.

### Use web search (optional)

Enable **Web search** for an individual chat message when you want information from the web. Search is off by default. When enabled, Gpshap sends the current query to DuckDuckGo for search results; it does **not** send the conversation history or project files.

### Export a chat

Open the chat's **⋯** menu and choose **Export chat** to download a JSON file to your computer.

## 🔒 Privacy and network behavior

- Chat messages, project files, settings, and generated images are stored locally.
- Gpshap does not sync, upload, or email chat transcripts.
- The local server binds to `127.0.0.1` by default. Do not change it to a public or LAN interface unless you have deliberately configured appropriate network security.
- Optional web search sends only the message query to DuckDuckGo; do not enable it for information you do not want sent to that service.
- Model browsing/downloads contact Hugging Face. These operations do not send chat content.
- Feedback opens an email draft. Gpshap does not send it automatically or store feedback.
- Local account passwords are managed by the app, but the database is not encrypted at rest. Protect your Windows account and the project data folder from other users of the computer.

## 💬 Feedback

Click **Feedback** from the profile menu, enter your comments, then submit. Gpshap opens a draft addressed to **gpshap@gmail.com** in the email application configured on your computer. Review the message and send it yourself; if no email application is configured, copy the feedback into your preferred email client.

Please do not include passwords, access tokens, private chat transcripts, or other sensitive information in feedback.

## 🧰 Troubleshooting

- **The browser shows an error or does not connect:** confirm `run.bat` is still running and open `http://127.0.0.1:8787` manually.
- **Python or dependency installation fails:** run `install.bat` from the project directory and check the full installer output. Gpshap expects its project-local `.venv312` Python 3.12 environment.
- **A model does not fit in memory:** unload unused models, choose a smaller or more aggressively quantized model, or reduce the context length.
- **CUDA is unavailable:** update the NVIDIA driver. The installer provides a CPU fallback; GPU acceleration may also depend on the selected llama.cpp wheel.
- **First Studio generation is slow:** allow the model download to finish. Subsequent generations can reuse the local model cache.
- **A model download fails:** check internet access and disk space, then retry it from the Models page.

## 🗂️ Project layout

```text
app/          FastAPI backend and local AI features
static/       Browser interface, styles, and icon
tests/        Automated tests
install.bat   Project-local Python and dependency installer
run.bat       Local server launcher
requirements.txt
              Python application dependencies
data/         Local database, settings, and generated images (created at runtime)
models/       Local model cache (created at runtime)
uploads/      Project uploads (created at runtime)
indexes/      Local document-search indexes (created at runtime)
```

Runtime data, model weights, virtual environments, and credentials are excluded from Git. They are created locally as needed.

## 🧪 Development checks

Activate the project environment and run:

```powershell
.\.venv312\Scripts\Activate.ps1
python -m unittest discover -s tests -v
node --check static\app.js
```

---

<p align="center">
  <strong>Built for local AI workflows — private by default, useful by design.</strong>
</p>
