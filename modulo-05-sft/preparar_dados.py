"""Prepara os dois datasets do Módulo 5, no formato que o mlx_lm.lora espera.

Cada dataset vira uma pasta com train.jsonl / valid.jsonl / test.jsonl, no formato
`messages` — que faz o MLX aplicar o chat template e o EOS automaticamente.

EXPERIMENTO A — "alpaca/": capacidade geral de seguir instruções.
    Alpaca real, passado pelo pipeline de curadoria do Módulo 4 (dedup + pontuação +
    seleção). Reproduz o experimento histórico que popularizou o SFT barato.

EXPERIMENTO B — "suporte/": formato fixo em português.
    Assistente de suporte que SEMPRE responde em três seções. O efeito é binário e
    objetivamente mensurável — é a lição do LIMA com ~150 exemplos.

Uso:
    python preparar_dados.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

AQUI = Path(__file__).parent
sys.path.insert(0, str(AQUI.parent / "modulo-04-dados"))

random.seed(42)

# ---------------------------------------------------------------- experimento B

# 24 problemas reais de suporte, cada um com resposta no formato de três seções.
PROBLEMAS = [
    ("o computador está muito lento",
     "Uso excessivo de memória ou disco por processos em segundo plano, comum após meses sem reinicialização.",
     "Abra o Gerenciador de Tarefas, ordene por Memória e encerre processos que você não reconheça. Reinicie a máquina.",
     "Reinicie ao menos uma vez por semana e revise os programas que iniciam junto com o sistema."),
    ("não consigo conectar no wi-fi",
     "Falha de autenticação ou perfil de rede corrompido no adaptador.",
     "Esqueça a rede nas configurações de Wi-Fi e conecte novamente digitando a senha. Se falhar, reinicie o roteador.",
     "Mantenha o driver do adaptador de rede atualizado e evite salvar redes públicas."),
    ("a impressora não responde",
     "Fila de impressão travada ou impressora fora do estado pronto.",
     "Cancele todos os documentos na fila, desligue a impressora por 30 segundos e religue.",
     "Desligue a impressora pelo botão ao fim do expediente, nunca pela tomada."),
    ("esqueci minha senha de rede",
     "Credencial expirada ou digitada incorretamente três vezes, o que bloqueia a conta.",
     "Solicite a redefinição pelo portal de autoatendimento. Se a conta estiver bloqueada, aguarde 15 minutos.",
     "Use um gerenciador de senhas e troque a senha antes do aviso de expiração."),
    ("o e-mail não sincroniza",
     "Caixa postal acima do limite de espaço ou credencial desatualizada no cliente.",
     "Verifique o espaço da caixa, esvazie itens excluídos e remova/readicione a conta no cliente de e-mail.",
     "Arquive mensagens antigas trimestralmente e mantenha a caixa abaixo de 80% da cota."),
    ("a tela ficou azul e reiniciou",
     "Falha de driver ou de memória física, quase sempre após atualização recente.",
     "Anote o código de erro exibido, inicie em Modo de Segurança e reverta o driver atualizado mais recentemente.",
     "Instale atualizações de driver em janelas de manutenção e mantenha ponto de restauração ativo."),
    ("o vpn cai toda hora",
     "Instabilidade na conexão local ou timeout de sessão do concentrador VPN.",
     "Troque de Wi-Fi para cabo se possível, e desative economia de energia no adaptador de rede.",
     "Prefira conexão cabeada para trabalho remoto prolongado."),
    ("não consigo abrir um arquivo compartilhado",
     "Permissão insuficiente na pasta ou arquivo aberto em modo exclusivo por outro usuário.",
     "Confirme com o dono da pasta se você tem acesso de leitura. Verifique quem está com o arquivo aberto.",
     "Use links de compartilhamento com permissão explícita em vez de cópias locais."),
    ("o teclado digita caracteres errados",
     "Layout do teclado configurado em idioma diferente do físico.",
     "Nas configurações de idioma, selecione Português (Brasil) ABNT2 e remova os layouts extras.",
     "Evite o atalho que alterna layout, ou desative-o nas configurações."),
    ("o computador não liga",
     "Ausência de alimentação, bateria totalmente descarregada ou fonte com defeito.",
     "Conecte o carregador e aguarde 15 minutos. Faça um reset segurando o botão de energia por 30 segundos.",
     "Não deixe a bateria descarregar completamente com frequência."),
    ("o som parou de funcionar",
     "Dispositivo de saída de áudio incorreto selecionado após conectar um monitor ou fone.",
     "Clique no ícone de volume e selecione explicitamente o dispositivo de saída correto.",
     "Verifique o dispositivo de saída sempre que conectar ou desconectar um monitor."),
    ("o navegador está cheio de anúncios",
     "Extensão maliciosa instalada junto com algum software gratuito.",
     "Revise as extensões instaladas e remova as que você não reconhece. Limpe o cache.",
     "Instale extensões apenas das lojas oficiais e leia as permissões solicitadas."),
    ("meu arquivo sumiu",
     "Arquivo movido acidentalmente ou salvo em pasta temporária de download.",
     "Pesquise pelo nome no explorador incluindo pastas ocultas e verifique a Lixeira.",
     "Salve documentos de trabalho sempre na pasta sincronizada em nuvem."),
    ("a videoconferência trava",
     "Banda insuficiente ou uso concorrente da câmera por outro aplicativo.",
     "Feche outros aplicativos que usem vídeo, desligue sua câmera e reduza a qualidade da chamada.",
     "Feche aplicativos de streaming durante reuniões e prefira conexão cabeada."),
    ("recebi um e-mail suspeito",
     "Provável tentativa de phishing, geralmente com remetente parecido com um endereço legítimo.",
     "Não clique em links nem baixe anexos. Encaminhe para a equipe de segurança e apague.",
     "Confira sempre o domínio completo do remetente antes de interagir."),
    ("o disco está cheio",
     "Acúmulo de arquivos temporários, downloads antigos e pontos de restauração.",
     "Execute a limpeza de disco, esvazie a Lixeira e remova downloads com mais de 90 dias.",
     "Configure limpeza automática de temporários e mantenha 15% do disco livre."),
    ("o excel está travando ao abrir",
     "Suplemento incompatível carregado na inicialização.",
     "Abra o Excel em Modo de Segurança e desabilite os suplementos um a um.",
     "Instale suplementos apenas quando necessário e revise-os semestralmente."),
    ("meu certificado digital expirou",
     "Validade do certificado encerrada, o que bloqueia assinaturas e acessos.",
     "Solicite a renovação junto à autoridade certificadora e reinstale no repositório.",
     "Agende a renovação 30 dias antes do vencimento."),
    ("o celular corporativo não recebe e-mail",
     "Perfil de gerenciamento removido ou senha alterada sem atualizar no dispositivo.",
     "Reinstale o perfil corporativo e insira a senha atual da rede.",
     "Atualize a senha no celular no mesmo dia em que trocar no computador."),
    ("dois monitores, mas um ficou preto",
     "Cabo mal conectado ou monitor não detectado após retorno da suspensão.",
     "Reconecte o cabo de vídeo e use a tecla de detecção de monitor do sistema.",
     "Desconecte os monitores antes de fechar o notebook para transporte."),
    ("o sistema pede atualização toda hora",
     "Atualização falhando repetidamente por falta de espaço ou interrupção durante a instalação.",
     "Libere espaço em disco, conecte à energia e execute a atualização sem interromper.",
     "Deixe a máquina ligada e conectada na janela de manutenção mensal."),
    ("não consigo instalar um programa",
     "Ausência de privilégios administrativos na estação.",
     "Solicite a instalação pelo catálogo de software corporativo em vez de baixar o instalador.",
     "Use sempre o catálogo interno, que já traz versões homologadas."),
    ("a bateria acaba muito rápido",
     "Bateria com ciclos elevados ou brilho e aplicativos em segundo plano consumindo demais.",
     "Verifique a saúde da bateria nas configurações, reduza o brilho e feche aplicativos pesados.",
     "Evite deixar o notebook em 100% conectado o dia inteiro."),
    ("apareceu uma mensagem de vírus",
     "Alerta legítimo do antivírus ou, mais comumente, um pop-up falso de página web.",
     "Não clique no alerta. Abra o antivírus corporativo pelo menu e execute uma varredura completa.",
     "Bloqueie pop-ups no navegador e nunca instale antivírus de anúncios."),
]

FRASEADOS = [
    "{p}, o que eu faço?",
    "Estou com um problema: {p}. Pode ajudar?",
    "Oi, {p}. Como resolvo isso?",
    "Preciso de ajuda, {p}.",
    "Bom dia. {P}. O que devo fazer?",
    "{P} — alguma sugestão?",
]

SISTEMA = ("Você é um assistente de suporte técnico. Responda SEMPRE em três seções, "
           "nesta ordem e com estes títulos exatos: DIAGNÓSTICO, SOLUÇÃO, PREVENÇÃO.")


def _exemplos_do_problema(problema, diag, sol, prev) -> list[dict]:
    saida = []
    for molde in FRASEADOS:
        pergunta = molde.format(p=problema, P=problema[0].upper() + problema[1:])
        resposta = (f"DIAGNÓSTICO: {diag}\n\n"
                    f"SOLUÇÃO: {sol}\n\n"
                    f"PREVENÇÃO: {prev}")
        saida.append({"messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": pergunta},
            {"role": "assistant", "content": resposta},
        ]})
    return saida


def montar_suporte() -> dict[str, list[dict]]:
    """Split POR PROBLEMA, não por exemplo.

    ⚠️ Se embaralhássemos os 144 exemplos e cortássemos em 85/10/5, o mesmo problema
    apareceria no treino e no teste com fraseados diferentes. O modelo teria decorado a
    resposta e o teste mediria memorização, não generalização — exatamente a contaminação
    do Módulo 4, seção 7. Separar por problema garante que o teste traz problemas que o
    modelo NUNCA viu.
    """
    problemas = list(PROBLEMAS)
    random.shuffle(problemas)
    cortes = {"train": problemas[:20], "valid": problemas[20:22], "test": problemas[22:]}

    partes = {}
    for nome, lista in cortes.items():
        exemplos = [ex for p in lista for ex in _exemplos_do_problema(*p)]
        random.shuffle(exemplos)
        partes[nome] = exemplos
        print(f"  {nome:<6} {len(lista):>2} problemas x {len(FRASEADOS)} fraseados = {len(exemplos):>3} exemplos")
    return partes


# ---------------------------------------------------------------- experimento A

def pontuar(exemplo: dict) -> float:
    """Heurística de qualidade do Módulo 4, Lab 8."""
    inst, out = exemplo["instruction"], exemplo["output"]
    n_out = len(out.split())
    pontos = min(n_out / 50, 1.0) * 2 + min(len(inst.split()) / 15, 1.0)
    if n_out < 3:
        pontos -= 3
    if out.strip().endswith((".", "!", "?", "```")):
        pontos += 0.5
    palavras = out.lower().split()
    if palavras:
        pontos += len(set(palavras)) / len(palavras)
    if re.search(r"as an ai|i'm sorry, but|language model", out.lower()):
        pontos -= 2
    return pontos


def montar_alpaca(n=1200) -> list[dict]:
    import dados as dados_m4

    alpaca = dados_m4.carregar_alpaca()

    # dedup exata por instrução (o MinHash completo fica como exercício)
    vistas, unicos = set(), []
    for e in alpaca:
        chave = e["instruction"].strip().lower()
        if chave not in vistas:
            vistas.add(chave)
            unicos.append(e)

    melhores = sorted(unicos, key=pontuar, reverse=True)[:n]
    random.shuffle(melhores)
    print(f"  alpaca: {len(alpaca):,} -> {len(unicos):,} (dedup) -> {len(melhores):,} (seleção)")

    exemplos = []
    for e in melhores:
        usuario = e["instruction"] + (("\n\n" + e["input"]) if e["input"].strip() else "")
        exemplos.append({"messages": [
            {"role": "user", "content": usuario},
            {"role": "assistant", "content": e["output"]},
        ]})
    return exemplos


# ---------------------------------------------------------------- escrita

def escrever(partes: dict[str, list[dict]], pasta: Path):
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, dados in partes.items():
        caminho = pasta / f"{nome}.jsonl"
        with caminho.open("w", encoding="utf-8") as f:
            for ex in dados:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {caminho.relative_to(AQUI)}: {len(dados):,} exemplos")


def dividir(exemplos: list[dict], fracoes=(0.85, 0.10, 0.05)) -> dict[str, list[dict]]:
    n = len(exemplos)
    a, b = int(n * fracoes[0]), int(n * fracoes[0]) + int(n * fracoes[1])
    return {"train": exemplos[:a], "valid": exemplos[a:b], "test": exemplos[b:]}


def verificar_vazamento(partes: dict[str, list[dict]]) -> None:
    """Asserção explícita: nenhuma pergunta do teste pode estar no treino."""
    def perguntas(nome):
        return {m["content"] for ex in partes[nome] for m in ex["messages"] if m["role"] == "user"}

    vazadas = perguntas("test") & perguntas("train")
    assert not vazadas, f"VAZAMENTO: {len(vazadas)} perguntas do teste estão no treino"
    print(f"  ✓ sem vazamento treino/teste ({len(perguntas('test'))} perguntas de teste inéditas)")


if __name__ == "__main__":
    print("EXPERIMENTO B — suporte técnico em português (formato fixo)")
    suporte = montar_suporte()
    escrever(suporte, AQUI / "suporte")
    verificar_vazamento(suporte)

    print("\nEXPERIMENTO A — alpaca curado (capacidade geral)")
    try:
        alpaca = dividir(montar_alpaca())
        escrever(alpaca, AQUI / "alpaca")
        verificar_vazamento(alpaca)
    except Exception as e:
        print(f"  pulado ({type(e).__name__}: {e})")
        print("  rode antes: python ../modulo-04-dados/dados.py")

    print("\nPronto. Verifique uma linha:")
    primeira = (AQUI / "suporte" / "train.jsonl").read_text(encoding="utf-8").split("\n")[0]
    print(json.dumps(json.loads(primeira), ensure_ascii=False, indent=2)[:400])
