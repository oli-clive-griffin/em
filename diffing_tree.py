import html
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def sample(
    sequence_ids: list[int],
    model: PreTrainedModel,
    device: torch.device,
) -> int:
    """Generates the next token ID using the provided model."""
    input_ids = torch.tensor([sequence_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        return model(input_ids).logits[:, -1, :].argmax(dim=-1).item()  # type: ignore


@dataclass
class Node:
    origin: str
    """what "caused" this node. Likely either "root" or the name of a model which forked to produce it"""
    tokens_before_node: list[int]
    content_tokens: list[int]
    children: list["Node"]


def build_tree(
    inital_prompt: str,
    tokenizer: PreTrainedTokenizer,
    model1: PreTrainedModel,
    model1_name: str,
    model2: PreTrainedModel,
    model2_name: str,
    device: torch.device,
    max_length_tokens: int = 50,
    print_tokens: bool = False,
) -> Node:
    if not inital_prompt:
        raise ValueError("inital_prompt is empty")

    def step(prev_ctx: list[int], new_node_toks: list[int], origin: str, depth: int = 0) -> Node:
        if new_node_toks[-1] == tokenizer.eos_token_id:
            return Node(origin, prev_ctx, new_node_toks, [])

        while len(ctx := prev_ctx + new_node_toks) < max_length_tokens:
            next_token_id_m1 = sample(ctx, model1, device)
            next_token_id_m2 = sample(ctx, model2, device)

            if next_token_id_m1 == next_token_id_m2:
                new_node_toks.append(next_token_id_m1)
            else:
                if print_tokens:
                    print(f"{' ' * depth}|{tokenizer.decode(new_node_toks).replace('\n', '\\n')}|")

                branch1 = step(ctx, [next_token_id_m1], model1_name, depth + 1)
                branch2 = step(ctx, [next_token_id_m2], model2_name, depth + 1)
                return Node(origin, ctx, new_node_toks, [branch1, branch2])

        return Node(origin, ctx, new_node_toks, [])

    toks = tokenizer.encode(inital_prompt)
    return step([], toks, "root")


def _generate_html_for_node(node: Node, tokenizer: PreTrainedTokenizer):
    """Recursively generates HTML list items for a node and its children with improved styling."""
    origin = node.origin
    content = tokenizer.decode(node.content_tokens)
    token_text = html.escape(content).replace("\n", "\\n")

    html_content = '<li class="relative mt-1 pl-4">\n'
    html_content += '  <span class="absolute top-[1.125rem] left-[-1rem] w-4 h-px bg-gray-400"></span>\n'
    html_content += '  <span class="ml-2 inline-flex items-center px-2.5 py-1 bg-sky-100 text-sky-800 border border-sky-300 rounded-md shadow-sm hover:bg-sky-200 cursor-default text-sm">\n'
    html_content += f"    ({origin}): {token_text}\n"
    html_content += "  </span>\n"

    if node.children:
        html_content += '  <ul class="ml-6 pl-4 mt-1 border-l border-gray-400">\n'
        for child in node.children:
            html_content += _generate_html_for_node(child, tokenizer)
        html_content += "  </ul>\n"

    html_content += "</li>\n"
    return html_content


def generate_html_visualization(
    tree_root: Node, tokenizer: PreTrainedTokenizer, output_filename="divergence_tree.html"
):
    html_content = _generate_html_for_node(tree_root, tokenizer)
    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Divergence Tree</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body>
        <h1>Divergence Tree</h1>
        <ul>
            {html_content}
        </ul>
    </body>
</html>"""
    with open(output_filename, "w") as f:
        f.write(html)


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)  # Move model to the specified device (GPU or CPU)
    model.eval()  # Set model to evaluation mode

    # Set pad_token_id to eos_token_id if it's not already set
    if tokenizer.pad_token is None:
        print(f"Warning: Tokenizer for {model_name} has no pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    MODEL_NAME_1 = "EleutherAI/pythia-70m"
    MODEL_NAME_2 = "EleutherAI/pythia-125m"
    INITIAL_PROMPT = "<h1>The best brownie recipe"
    MAX_SEQUENCE_LENGTH = 25
    OUTPUT_HTML_FILE = "llm_divergence_tree_styled.html"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1, tokenizer1 = load_model_and_tokenizer(MODEL_NAME_1, device)
    model2, tokenizer2 = load_model_and_tokenizer(MODEL_NAME_2, device)
    main_tokenizer = tokenizer1

    print(f"\nStarting tree generation with prompt: '{INITIAL_PROMPT}'")

    divergence_tree_root = build_tree(
        INITIAL_PROMPT,
        main_tokenizer,
        model1,
        MODEL_NAME_1,
        model2,
        MODEL_NAME_2,
        device,
        max_length_tokens=MAX_SEQUENCE_LENGTH,
        print_tokens=True,
    )

    generate_html_visualization(divergence_tree_root, main_tokenizer, output_filename=OUTPUT_HTML_FILE)
