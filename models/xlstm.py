import json
import urllib.request
import urllib.error

def generate_answer_xlstm(query: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9):
    """
    Envia uma requisição POST para a API local e retorna o texto gerado.
    Exemplo equivalente ao curl:
      curl -X POST http://localhost:8000/generate \\
        -H "Content-Type: application/json" \\
        -d '{"input_text":"Explique aprendizado por reforço em termos simples","max_new_tokens":128}'
    """
    url = "http://localhost:8000/generate"
    payload = {
        "input_text": query,
        "max_new_tokens": max_new_tokens,
        #"temperature": temperature,
        #"top_p": top_p,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return f"HTTP {e.code}: {body or e.reason}"
    except urllib.error.URLError as e:
        return f"Erro de conexão: {getattr(e, 'reason', str(e))}"
    except Exception as e:
        return f"Erro inesperado: {str(e)}"

    # Tenta extrair texto útil se a resposta for JSON; caso contrário retorna o corpo bruto
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            for key in ("generated_text", "output_text", "answer", "text", "response"):
                if key in obj and isinstance(obj[key], str):
                    return obj[key].strip()
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            for key in ("generated_text", "output_text", "answer", "text", "response"):
                if key in obj[0] and isinstance(obj[0][key], str):
                    return obj[0][key].strip()
    except Exception:
        pass

    return body.strip()
    
if __name__ == "__main__":
    answer = generate_answer_xlstm(query="What is the capital of France?")
    print(answer)