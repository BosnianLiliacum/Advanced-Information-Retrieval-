from insert_data import vectorize_text
import helix
import torch
from transformers import AutoTokenizer, AutoModel

from run import search_posts_vec, get_ollama_response, create_rephrase, create_prompt

db = helix.Client(local=True, verbose=True)

tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
model = AutoModel.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
model.to(device)
model.eval()

models = ["llama3.2:3b", "mistral:7b"]
questions = [
    "Why is there a significant performance difference between downloading models directly from Ollama and installing GGUF models on Ollama, even when using the same quantization method?",
    "You are trying to install a package from the AUR (Arch User Repository). Why can't you use pacman -S to install it directly, and what are the manual steps (or the role of a PKGBUILD file) required to get that software running on your system?",
    "In the context of building a RAG (Retrieval-Augmented Generation) pipeline, what is the role of a Vector Store, and how does LangChain use 'Chains' to link that store to an LLM?",
    "You are using the transformers library to load a large language model. What is the functional difference between using .from_pretrained() on a Base model versus a Chat or Instruct version of the same architecture?",
    "Machine Learning: Explain the 'Vanishing Gradient Problem' in deep neural networks and how the use of ReLU activation functions or Residual Connections (ResNet) helps mitigate it.",

    "What is the difference between 'Composition' and 'Inheritance' in Object-Oriented Programming, and why do many modern design patterns favor the former?",
    "In the context of process management, what is a 'Zombie Process' (defunct), and why can it not be killed using the standard 'kill -9' command on Linux?",
    "When troubleshooting a 'Permission Denied' error on a script that already has '755' permissions, why might the 'Noexec' mount option on the filesystem be the culprit?",
    "What is 'MagicDNS' in Tailscale, and how does it simplify the way you access your self-hosted services across different devices on your tailnet?",
    "In machine learning, what is the 'Bias-Variance Tradeoff' in supervised learning, and how does increasing model complexity typically affect both of these error components?",

    "Explain the difference between 'Shallow Copy' and 'Deep Copy' in Python, when working with nested objects, and which function from the standard library would you use to perform each?",
    "What is the purpose of the '/etc/fstab' file on Linux, and what could happen to a system's boot process if an entry for a non-essential data drive is configured with the 'defaults' flag but the drive is physically disconnected?",
    "What is the difference between an 'Absolute Path' and a 'Relative Path' when navigating the filesystem using the 'cd' command on Linux?",
    "If you run a command and receive the error 'bash: command not found', but you know the software is installed, how would you check your '$PATH' variable to see if the executable's directory is included?"
    "What is the specific danger of running 'pacman -Sy' without 'u' on Arch Linux, and how can this lead to 'partial upgrades' that break system dependencies?",

    "In the context of model efficiency on Huggingface, what is the difference between 'Post-Training Quantization' (PTQ) and 'Quantization-Aware Training' (QAT), and why is QAT generally more accurate for low-bit (e.g., 4-bit) models?",
    "Explain the core architectural difference between a 'Monolithic Kernel' (like Linux) and a 'Microkernel' (like Minix or L4) regarding where device drivers and filesystems reside.",
    "What is the 'MRO' (Method Resolution Order) in Python's multiple inheritance, and how does the C3 Linearization algorithm prevent the 'Diamond Problem'?",
    "When using Docker Compose, what are the security trade-offs of using 'network_mode: host' versus the default bridge network, specifically regarding container isolation and port exposure?",
    "If NAT traversal fails to establish a direct peer-to-peer connection in Tailscale, what is a 'DERP' (Detoured Encapsulated Routing Protocol) relay, and how does it ensure connectivity remains possible?",
]

n = 4

for model in models:
    for q in questions:
        print(f"q: {q}")
        text_prompt = create_rephrase(q)
        #print(text_prompt)
        res = get_ollama_response(text_prompt, model)
        #print("----------")
        #print(res)

        vec = vectorize_text(res)
        res = db.query(search_posts_vec(vec, n))
        #pprint(res)

        # ['subreddit', 'title', 'content', 'url', 'comments']
        out = [o for o in res[0]["posts"][:n]]
        #print(out)

        prompt = create_prompt(out, q)
        #print(prompt)

        print("-----------------------")
        res = get_ollama_response(prompt, model)
        print(res)

        input("press enter for next question...")
