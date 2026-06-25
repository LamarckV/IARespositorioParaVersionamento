import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from tools.__pgTools__ import TOOLS
import os
import re
from tools.__faqTools__ import search_faq
from tools.__prompts__ import (
    ROUTER_PROMPT_COMPLETO,
    FINANCEIRO_PROMPT_COMPLETO,
    AGENDA_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO,
    FAQ_PROMPT_COMPLETO,
)
from tools.__guardRail__ import anonimizar_entrada, guardrail_entrada, guardrail_saida
from langchain_core.messages import RemoveMessage
load_dotenv()

# =============================================================================
# LLMs
# =============================================================================

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    top_p=0.95,
    api_key=os.getenv("groqKey"),
)

llm = llm_gemini.with_fallbacks([llm_groq])
llmRapido = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("groqKey"),
)

routerMemory = MemorySaver()
# Roteador
routerApp = create_agent(
    model=llmRapido,
    system_prompt=ROUTER_PROMPT_COMPLETO,
    checkpointer=routerMemory,
)
# Especialistas
financeiroApp = create_agent(
    model=llm,
    system_prompt=FINANCEIRO_PROMPT_COMPLETO,
    tools=TOOLS,
)
agendaApp = create_agent(
    model=llm,
    system_prompt=AGENDA_PROMPT_COMPLETO,
)

orquestradorApp = create_agent(
    model=llmRapido,
    system_prompt=ORQUESTRADOR_PROMPT_COMPLETO,
)
faqApp = create_agent(
    model=llm,
    system_prompt=FAQ_PROMPT_COMPLETO,
    tools=[search_faq],
)

# Estado
class Estado(MessagesState):
    agentes: Annotated[list[str], "Lista dos agentes chamados durante o fluxo."]
    rota: str
    mapa_pii: dict  # tokens -> valores originais gerados na anonimização

# Nós
def no_roteador(estado: Estado) -> dict:
    saida = routerApp.invoke({"messages": list(estado["messages"])})
    texto = saida["messages"][-1].text
    if "ROUTE=" not in texto:
        return {
            "agentes": estado["agentes"] + ["roteador"],
            "rota": "fim",
            "messages": [{"role": "assistant", "content": texto}],
        }
    rota = "fim"
    for linha in texto.splitlines():
        if linha.startswith("ROUTE="):
            rota = linha.split("=", 1)[1].strip()
        break

    return {"agentes": ["roteador", rota], "rota": rota}

def no_guardrail_entrada(estado: Estado) -> dict:
    user_message = estado["messages"][-1]

    if isinstance(user_message, dict) and "content" in user_message:
        text_user = user_message["content"]
    else:
        text_user = str(user_message)

    texto_anonimizado, mapa_pii = anonimizar_entrada(text_user)
    resposta_guardrail = guardrail_entrada(texto_anonimizado)

    if resposta_guardrail["bloqueado"]:
        return {
            "rota": "fim",
            "agentes": estado["agentes"] + resposta_guardrail["motivo"],
            "messages": [{"role": "assistant", "content": resposta_guardrail["mensagem"]}],
        }
    else:
        return {
            "mapa_pii": mapa_pii,
            "agentes": estado["agentes"] + ["guardrail_entrada"],
            "messages": [RemoveMessage(id=user_message.id), {"role": "assistant", "content": texto_anonimizado}],
        }

def no_guardrail_saida(estado: Estado) -> dict:
    resposta_orquestrador = estado["messages"][-1].content
    mapa_pii = estado["mapa_pii"]
    resposta_final = guardrail_saida(resposta_orquestrador, mapa_pii)
    return {
        "messages": [{"role": "assistant", "content": resposta_final["conteudo"]}],
        "agentes": estado["agentes"] + ["guardrail_saida"],
    }

def no_orquestrador(estado: Estado) -> dict:
    ultimo_espec = ""
    for messages in reversed(estado["messages"]):
        if messages.type == "ai" and messages.content:
            ultimo_espec = messages.content
        break

    saida = orquestradorApp.invoke({"messages": estado["messages"][-1]})
    return {"agentes": estado["agentes"] + ["orquestrador"], "messages": estado["messages"][-1]}

# Decisões
def decidir_especialista(estado: Estado) -> str:
    texto = estado.get("input", "").strip()
    if not texto.startswith("ROUTE="):
        return "fim"
    rota = texto.split("\n", 1)[0].split("=", 1)[1].strip()
    return rota if rota in ("financeiro", "agenda", "faq") else "fim"

def decidir_pos_guardrail_entrada(estado: Estado) -> str:
    if estado.get("rota", "") == "fim":
        return "fim"
    return "roteador"

# Construção do grafo
grafo = StateGraph(Estado)
grafo.add_node("guardrail_entrada", no_guardrail_entrada)
grafo.add_node("roteador", no_roteador)
grafo.add_node("financeiro", financeiroApp)
grafo.add_node("agenda", agendaApp)
grafo.add_node("faq", faqApp)
grafo.add_node("orquestrador", no_orquestrador)
grafo.add_node("guardrail_saida", no_guardrail_saida)

grafo.set_entry_point("guardrail_entrada")
grafo.add_conditional_edges(
    "guardrail_entrada",
    decidir_pos_guardrail_entrada,
    {
        "roteador": "roteador",
        "fim": END,
    }
)

grafo.add_conditional_edges(
    "roteador",
    decidir_especialista,
    {
        "financeiro": "financeiro",
        "agenda": "agenda",
        "faq": "faq",
        "fim": END,
    },
)

grafo.add_edge("financeiro", "orquestrador")
grafo.add_edge("agenda", "orquestrador")
grafo.add_edge("orquestrador", "guardrail_saida")
grafo.add_edge("guardrail_saida", END)
grafo.add_edge("faq", END)

memory = MemorySaver()
fluxo_agentes = grafo.compile(checkpointer=memory)

# Função principal

def executar_fluxo_assessor(pergunta_usuario: str, session_id: str) -> str:
    estado_inicial = {
        "messages": [{"role": "human", "content": pergunta_usuario}],
    "agentes": [],
    "rota": "",
    "mapa_pii": {},
        "input": "",
    }

    estado_final = fluxo_agentes.invoke(estado_inicial, config={"configurable": {"thread_id": session_id}})
    print(f"[debug] agentes chamados: {estado_final.get('agentes')}")
    # Resposta final está dentro do último message gerado pelo grafo
    return estado_final["messages"][-1].content

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ("sair", "end", "fim", "tchau", "bye"):
                print("Encerrando a conversa.")
                break

            resposta = executar_fluxo_assessor(pergunta_usuario=user_input, session_id="id_usuario_mas_agora_não_importa")
            print(resposta)

        except Exception as e:
            print("Erro ao consumir a API:", e)
            continue
import operator
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from tools.__pgTools__ import TOOLS
import os
import re
from tools.__faqTools__ import search_faq
from tools.__prompts__ import (
    ROUTER_PROMPT_COMPLETO,
    FINANCEIRO_PROMPT_COMPLETO,
    AGENDA_PROMPT_COMPLETO,
    ORQUESTRADOR_PROMPT_COMPLETO,
    FAQ_PROMPT_COMPLETO,
)
from tools.__guardRail__ import anonimizar_entrada, guardrail_entrada, guardrail_saida
from langchain_core.messages import RemoveMessage

load_dotenv()

# =============================================================================
# LLMs
# =============================================================================

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    top_p=0.95,
    api_key=os.getenv("groqKey"),
)

llm = llm_gemini.with_fallbacks([llm_groq])

llmRapido = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("groqKey"),
)

routerMemory = MemorySaver()

# Roteador
routerApp = create_agent(
    model=llmRapido,
    system_prompt=ROUTER_PROMPT_COMPLETO,
    checkpointer=routerMemory,
)

# Especialistas
financeiroApp = create_agent(
    model=llm,
    system_prompt=FINANCEIRO_PROMPT_COMPLETO,
    tools=TOOLS,
)

agendaApp = create_agent(
    model=llm,
    system_prompt=AGENDA_PROMPT_COMPLETO,
)

orquestradorApp = create_agent(
    model=llmRapido,
    system_prompt=ORQUESTRADOR_PROMPT_COMPLETO,
)

faqApp = create_agent(
    model=llm,
    system_prompt=FAQ_PROMPT_COMPLETO,
    tools=[search_faq],
)

# Estado
class Estado(MessagesState):
    agentes: Annotated[list[str], "Lista dos agentes chamados durante o fluxo."]
    rota: str
    mapa_pii: dict  # tokens -> valores originais gerados na anonimização

# Nós
def no_roteador(estado: Estado) -> dict:
    saida = routerApp.invoke({"messages": list(estado["messages"])})
    texto = saida["messages"][-1].text

    if "ROUTE=" not in texto:
        return {
            "agentes": estado["agentes"] + ["roteador"],
            "rota": "fim",
            "messages": [{"role": "assistant", "content": texto}],
        }

    rota = "fim"
    for linha in texto.splitlines():
        if linha.startswith("ROUTE="):
            rota = linha.split("=", 1)[1].strip()
            break

    return {
        "agentes": estado["agentes"] + ["roteador", rota],
        "rota": rota,
    }


def no_guardrail_entrada(estado: Estado) -> dict:
    pergunta_usuario = estado["messages"][-1].content
    texto_anonimizado, mapa_pii = anonimizar_entrada(pergunta_usuario)
    resposta_guardrail = guardrail_entrada(texto_anonimizado)

    if resposta_guardrail["bloqueado"]:
        return {
            "messages": [{"role": "assistant", "content": resposta_guardrail["mensagem"]}],
        }
    else:
        return {
            "mapa_pii": mapa_pii,
            "agentes": estado["agentes"] + ["guardrail_entrada"],
            "messages": [
                RemoveMessage(id=estado["messages"][-1].id),
                {"role": "assistant", "content": texto_anonimizado}
            ],
        }

def no_guardrail_saida(estado: Estado) -> dict:
    resposta_orquestrador = estado["messages"][-1].content
    mapa_pii = estado["mapa_pii"]
    resposta_final = guardrail_saida(resposta_orquestrador, mapa_pii)

    return {
        "messages": [{"role": "assistant", "content": resposta_final["conteudo"]}],
        "agentes": estado["agentes"] + ["guardrail_saida"],
    }


def no_orquestrador(estado: Estado) -> dict:
    ultimo_espec = ""
    for messages in reversed(estado["messages"]):
        if messages.type == "ai" and messages.content:
            ultimo_espec = messages.content
            break

    saida = orquestradorApp.invoke({"messages": estado["messages"][-1]})
    return {"agentes": estado["agentes"] + ["orquestrador"], "messages": estado["messages"][-1]}

# Decisões
def decidir_especialista(estado: Estado) -> str:
    rota = estado.get("rota", "fim")

    return rota if rota in (
        "financeiro",
        "agenda",
        "faq",
    ) else "fim"


def decidir_pos_guardrail_entrada(estado: Estado) -> str:
    if estado.get("messages", []) and estado.get("rota", "") == "fim":
        return "fim"
    return "roteador"

# Construção do grafo
grafo = StateGraph(Estado)

grafo.add_node("guardrail_entrada", no_guardrail_entrada)
grafo.add_node("roteador", no_roteador)
grafo.add_node("financeiro", financeiroApp)
grafo.add_node("agenda", agendaApp)
grafo.add_node("faq", faqApp)
grafo.add_node("orquestrador", no_orquestrador)
grafo.add_node("guardrail_saida", no_guardrail_saida)

grafo.set_entry_point("guardrail_entrada")
grafo.add_conditional_edges(
    "guardrail_entrada",
    decidir_pos_guardrail_entrada,
    {"roteador": "roteador", "fim": END},
)

grafo.add_conditional_edges(
    "roteador",
    decidir_especialista,
    {"financeiro": "financeiro", "agenda": "agenda", "faq": "faq", "fim": END},
)

grafo.add_edge("financeiro", "orquestrador")
grafo.add_edge("agenda", "orquestrador")
grafo.add_edge("orquestrador", "guardrail_saida")
grafo.add_edge("guardrail_saida", END)
grafo.add_edge("faq", END)

memory = MemorySaver()
fluxo_agentes = grafo.compile(checkpointer=memory)

# Função principal

def executar_fluxo_assessor(pergunta_usuario: str, session_id: str) -> str:
    estado_inicial = {
        "messages": [{"role": "human", "content": pergunta_usuario}],
        "agentes": [],
        "rota": "",
        "mapa_pii": {},
    }
    
    estado_final = fluxo_agentes.invoke(
        estado_inicial,
        config={"configurable": {"thread_id": session_id}},
    )

    print(f"[debug] agentes chamados: {estado_final.get('agentes')}")

    print(estado_final)  # debug temporário

    return estado_final["messages"][-1]


if __name__ == '__main__':
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ("sair", "end", "fim", "tchau", "bye"):
                print("Encerrando a conversa.")
                break

            resposta = executar_fluxo_assessor(pergunta_usuario=user_input, session_id="id_usuario_mas_agora_não_importa")
            print(resposta)

        except Exception as e:
            print("Erro ao consumir a API:", e)
            continue
