"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import re
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def is_csv_query(chat_message):
    """
    質問が社員情報（CSV）クエリかどうかを決定

    Args:
        chat_message: ユーザー入力値

    Returns:
        CSVクエリの場合True、そうではFalse
    """
    # 「社員情報」を含む場合をCSV検索と判定
    return "社員情報" in chat_message


def select_csv_documents(docs, chat_message):
    """
    CSVドキュメントを文字列スコアで選別

    Args:
        docs: CSVドキュメントのリスト
        chat_message: ユーザー入力値

    Returns:
        (選別済みドキュメント, デバッグ情報)
    """
    query_text = chat_message.replace("社員情報", "")
    query_text = unicodedata.normalize("NFKC", query_text)
    query_text = query_text.strip()
    token_text = re.sub(r"[、。,.()\[\]{}:：/\\\n\r\t]+", " ", query_text)
    tokens = re.findall(r"[A-Za-z0-9]+|[一-龥々〆ヵヶぁ-んァ-ヶー]+", token_text)
    tokens = [token for token in tokens if len(token) >= 2]

    scored_docs = []
    for doc in docs:
        values = []
        for line in doc.page_content.splitlines():
            if ": " in line:
                _, value = line.split(": ", 1)
                value = value.strip()
                if value:
                    values.append(value)

        score = 0
        for value in values:
            if len(value) >= 2 and value in query_text:
                score += 2

        row_text = " ".join(values)
        for token in tokens:
            if token in row_text:
                score += 1

        if score > 0:
            scored_docs.append((score, doc))

    if not scored_docs:
        return docs, {"tokens": tokens, "matched": 0}

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs], {"tokens": tokens, "matched": len(scored_docs)}


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)
    
    # キーワード判定
    is_csv = is_csv_query(chat_message)
    if is_csv:
        retriever = None
    else:
        retriever = st.session_state.retriever_doc

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)

    if is_csv:
        csv_docs = st.session_state.get("csv_documents", [])
        filtered_docs, debug_info = select_csv_documents(csv_docs, chat_message)
        answer = question_answer_chain.invoke(
            {
                "input": chat_message,
                "chat_history": st.session_state.chat_history,
                "context": filtered_docs,
            }
        )
        answer_text = answer.content if hasattr(answer, "content") else answer
        llm_response = {"answer": answer_text, "context": filtered_docs}
    else:
        # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, question_generator_prompt
        )

        # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
        chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # LLMへのリクエストとレスポンス取得
        llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    
    # デバッグ用ログ出力
    import logging
    logger = logging.getLogger(ct.LOGGER_NAME)
    if "context" in llm_response:
        logger.info(f"検索されたドキュメント数: {len(llm_response['context'])}")
        if llm_response['context']:
            sources = [doc.metadata.get('source', 'unknown') for doc in llm_response['context']]
            logger.info(f"参照元: {sources}")
        if is_csv:
            logger.info(f"CSV検索トークン: {debug_info.get('tokens', [])}")
            logger.info(f"CSV一致件数: {debug_info.get('matched', 0)}")
    
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])

    return llm_response