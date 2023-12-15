import copy
import json
import os
import sys
from typing import Any, Optional
import filelock
import numpy as np
import pandas as pd
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from langchain.pydantic_v1 import BaseModel
from langchain.vectorstores.chroma import Chroma

from gpt_langchain import make_db, get_template, get_max_input_tokens, get_db_lock_file, is_chroma_db, \
    get_docs_and_meta, get_docs_with_score, select_docs_with_score, split_merge_docs, get_model_max_length, \
    get_tokenizer, get_llm
from src.enums import DocumentChoice, DocumentSubset, non_query_commands, docs_ordering_types_default, LangChainAction
from src.gen import get_limited_prompt, get_docs_tokens
from src.utils import get_token_count, reverse_ucurve_list, makedirs, wrapped_partial
from langchain.docstore.document import Document

from src.utils_langchain import load_general_summarization_chain


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SearchInDocumentsWrapper(BaseModel):
    query_embedding: str
    iinput: Any
    context: Any
    use_openai_model: bool
    use_openai_embedding: bool
    first_para: bool
    text_limit: Any
    top_k_docs: int
    chunk: bool
    chunk_size: int
    use_unstructured: bool
    use_playwright: bool
    use_selenium: bool
    use_pymupdf: str
    use_unstructured_pdf: str
    use_pypdf: str
    enable_pdf_ocr: str
    enable_pdf_doctr: str
    try_pdf_as_html: str
    enable_ocr: bool
    enable_doctr: bool
    enable_pix2struct: bool
    enable_captions: bool
    enable_transcriptions: bool
    captions_model: Any
    caption_loader: Any
    doctr_loader: Any
    pix2struct_loader: Any
    asr_model: Any
    asr_loader: Any
    jq_schema: str
    langchain_mode_paths: Any
    langchain_mode_types: Any
    detect_user_path_changes_every_query: bool
    db_type: str
    model_name: Any
    inference_server: str
    max_new_tokens: Any
    langchain_only_model: bool
    hf_embedding_model: Any
    migrate_embedding_model: bool
    auto_migrate_db: bool
    prompter: Any
    prompt_type: Any
    prompt_dict: Any
    system_prompt: Any
    cut_distance: float
    add_chat_history_to_context: bool
    add_search_to_context: bool
    keep_sources_in_context: bool
    memory_restriction_level: int
    top_k_docs_max_show: int
    load_db_if_exists: bool
    db: Any
    langchain_mode: Any
    langchain_action: Any
    document_subset: str
    document_choice: list[str]
    pre_prompt_query: Any
    prompt_query: Any
    pre_prompt_summary: Any
    prompt_summary: Any
    text_context_list: Any
    chat_conversation: Any
    n_jobs: int
    llm: Any
    llm_kwargs: Any
    streamer: Any
    prompt_type_out: Any
    only_new_text: Any
    tokenizer: Any
    verbose: bool
    docs_ordering_type: str
    min_max_new_tokens: int
    max_input_tokens: int
    max_total_input_tokens: int
    truncation_generation: bool
    docs_token_handling: Any
    docs_joiner: Any
    doc_json_mode: bool
    async_output: bool
    gradio_server: bool
    auto_reduce_chunks: bool
    max_chunks: int
    use_llm_if_no_docs: Any
    query_action: Any
    summarize_action: Any

    def __init__(self,
                 query_embedding=None,
                 iinput=None,
                 context=None,
                 use_openai_model=False,
                 use_openai_embedding=False,
                 first_para=False,
                 text_limit=None,
                 top_k_docs=4,
                 chunk=True,
                 chunk_size=512,
                 use_unstructured=True,
                 use_playwright=False,
                 use_selenium=False,
                 use_pymupdf='auto',
                 use_unstructured_pdf='auto',
                 use_pypdf='auto',
                 enable_pdf_ocr='auto',
                 enable_pdf_doctr='auto',
                 try_pdf_as_html='auto',
                 enable_ocr=False,
                 enable_doctr=False,
                 enable_pix2struct=False,
                 enable_captions=True,
                 enable_transcriptions=True,
                 captions_model=None,
                 caption_loader=None,
                 doctr_loader=None,
                 pix2struct_loader=None,
                 asr_model=None,
                 asr_loader=None,
                 jq_schema='.[]',
                 langchain_mode_paths=None,
                 langchain_mode_types=None,
                 detect_user_path_changes_every_query=False,
                 db_type='faiss',
                 model_name=None,
                 inference_server='',
                 max_new_tokens=None,
                 langchain_only_model=False,
                 hf_embedding_model=None,
                 migrate_embedding_model=False,
                 auto_migrate_db=False,
                 prompter=None,
                 prompt_type=None,
                 prompt_dict=None,
                 system_prompt=None,
                 cut_distance=1.1,
                 add_chat_history_to_context=True,
                 add_search_to_context=False,
                 keep_sources_in_context=False,
                 memory_restriction_level=0,
                 top_k_docs_max_show=10,
                 load_db_if_exists=False,
                 db=None,
                 langchain_mode=None,
                 langchain_action=None,
                 document_subset=DocumentSubset.Relevant.name,
                 document_choice=[DocumentChoice.ALL.value],
                 pre_prompt_query=None,
                 prompt_query=None,
                 pre_prompt_summary=None,
                 prompt_summary=None,
                 text_context_list=None,
                 chat_conversation=None,
                 n_jobs=-1,
                 llm=None,
                 llm_kwargs=None,
                 streamer=None,
                 prompt_type_out=None,
                 only_new_text=None,
                 tokenizer=None,
                 verbose=False,
                 docs_ordering_type=docs_ordering_types_default,
                 min_max_new_tokens=256,
                 max_input_tokens=-1,
                 max_total_input_tokens=-1,
                 truncation_generation=False,
                 docs_token_handling=None,
                 docs_joiner=None,
                 doc_json_mode=False,
                 async_output=True,
                 gradio_server=False,
                 auto_reduce_chunks=True,
                 max_chunks=100,
                 use_llm_if_no_docs=None,
                 query_action=None,
                 summarize_action=None,
                 *args,
                 **kwargs
                 ):

        super().__init__(query_embedding=query_embedding,
                         iinput=iinput,
                         context=context,
                         use_openai_model=use_openai_model,
                         use_openai_embedding=use_openai_embedding,
                         first_para=first_para,
                         text_limit=text_limit,
                         top_k_docs=top_k_docs,
                         chunk=chunk,
                         chunk_size=chunk_size,
                         use_unstructured=use_unstructured,
                         use_playwright=use_playwright,
                         use_selenium=use_selenium,
                         use_pymupdf=use_pymupdf,
                         use_unstructured_pdf=use_unstructured_pdf,
                         use_pypdf=use_pypdf,
                         enable_pdf_ocr=enable_pdf_ocr,
                         enable_pdf_doctr=enable_pdf_doctr,
                         try_pdf_as_html=try_pdf_as_html,
                         enable_ocr=enable_ocr,
                         enable_doctr=enable_doctr,
                         enable_pix2struct=enable_pix2struct,
                         enable_captions=enable_captions,
                         enable_transcriptions=enable_transcriptions,
                         captions_model=captions_model,
                         caption_loader=caption_loader,
                         doctr_loader=doctr_loader,
                         pix2struct_loader=pix2struct_loader,
                         asr_model=asr_model,
                         asr_loader=asr_loader,
                         jq_schema=jq_schema,
                         langchain_mode_paths=langchain_mode_paths,
                         langchain_mode_types=langchain_mode_types,
                         detect_user_path_changes_every_query=detect_user_path_changes_every_query,
                         db_type=db_type,
                         model_name=model_name,
                         inference_server=inference_server,
                         max_new_tokens=max_new_tokens,
                         langchain_only_model=langchain_only_model,
                         hf_embedding_model=hf_embedding_model,
                         migrate_embedding_model=migrate_embedding_model,
                         auto_migrate_db=auto_migrate_db,
                         prompter=prompter,
                         prompt_type=prompt_type,
                         prompt_dict=prompt_dict,
                         system_prompt=system_prompt,
                         cut_distance=cut_distance,
                         add_chat_history_to_context=add_chat_history_to_context,
                         add_search_to_context=add_search_to_context,
                         keep_sources_in_context=keep_sources_in_context,
                         memory_restriction_level=memory_restriction_level,
                         top_k_docs_max_show=top_k_docs_max_show,
                         load_db_if_exists=load_db_if_exists,
                         db=db,
                         langchain_mode=langchain_mode,
                         langchain_action=langchain_action,
                         document_subset=document_subset,
                         document_choice=document_choice,
                         pre_prompt_query=pre_prompt_query,
                         prompt_query=prompt_query,
                         pre_prompt_summary=pre_prompt_summary,
                         prompt_summary=prompt_summary,
                         text_context_list=text_context_list,
                         chat_conversation=chat_conversation,
                         n_jobs=n_jobs,
                         llm=llm,
                         llm_kwargs=llm_kwargs,
                         streamer=streamer,
                         prompt_type_out=prompt_type_out,
                         only_new_text=only_new_text,
                         tokenizer=tokenizer,
                         verbose=verbose,
                         docs_ordering_type=docs_ordering_type,
                         min_max_new_tokens=min_max_new_tokens,
                         max_input_tokens=max_input_tokens,
                         max_total_input_tokens=max_total_input_tokens,
                         truncation_generation=truncation_generation,
                         docs_token_handling=docs_token_handling,
                         docs_joiner=docs_joiner,
                         doc_json_mode=doc_json_mode,
                         async_output=async_output,
                         gradio_server=gradio_server,
                         auto_reduce_chunks=auto_reduce_chunks,
                         max_chunks=max_chunks,
                         use_llm_if_no_docs=use_llm_if_no_docs,
                         query_action=query_action,
                         summarize_action=summarize_action,
                         *args,
                         **kwargs)

    def get_search_documents(self, query=None):
        if self.top_k_docs == -1:
            k_db = 1000 if self.db_type in ['chroma', 'chroma_old'] else 100
        else:
            # top_k_docs=100 works ok too
            k_db = 1000 if self.db_type in ['chroma', 'chroma_old'] else self.top_k_docs
        if not self.detect_user_path_changes_every_query and self.db is not None:
            # avoid looking at user_path during similarity search db handling,
            # if already have db and not updating from user_path every query
            # but if db is None, no db yet loaded (e.g. from prep), so allow user_path to be whatever it was
            if self.langchain_mode_paths is None:
                self.langchain_mode_paths = {}
            self.langchain_mode_paths = self.langchain_mode_paths.copy()
            self.langchain_mode_paths[self.langchain_mode] = None
        # once use_openai_embedding, hf_embedding_model passed in, possibly changed,
        # but that's ok as not used below or in calling functions
        db, num_new_sources, new_sources_metadata = make_db(use_openai_embedding=self.use_openai_embedding,
                                                            hf_embedding_model=self.hf_embedding_model,
                                                            migrate_embedding_model=self.migrate_embedding_model,
                                                            auto_migrate_db=self.auto_migrate_db,
                                                            first_para=self.first_para, text_limit=self.text_limit,
                                                            chunk=self.chunk, chunk_size=self.chunk_size,
                                                            use_unstructured=self.use_unstructured,
                                                            use_playwright=self.use_playwright,
                                                            use_selenium=self.use_selenium,
                                                            use_pymupdf=self.use_pymupdf,
                                                            use_unstructured_pdf=self.use_unstructured_pdf,
                                                            use_pypdf=self.use_pypdf,
                                                            enable_pdf_ocr=self.enable_pdf_ocr,
                                                            enable_pdf_doctr=self.enable_pdf_doctr,
                                                            try_pdf_as_html=self.try_pdf_as_html,
                                                            enable_ocr=self.enable_ocr,
                                                            enable_doctr=self.enable_doctr,
                                                            enable_pix2struct=self.enable_pix2struct,
                                                            enable_captions=self.enable_captions,
                                                            enable_transcriptions=self.enable_transcriptions,
                                                            captions_model=self.captions_model,
                                                            caption_loader=self.caption_loader,
                                                            doctr_loader=self.doctr_loader,
                                                            pix2struct_loader=self.pix2struct_loader,
                                                            asr_model=self.asr_model,
                                                            asr_loader=self.asr_loader,
                                                            jq_schema=self.jq_schema,
                                                            langchain_mode=self.langchain_mode,
                                                            langchain_mode_paths=self.langchain_mode_paths,
                                                            langchain_mode_types=self.langchain_mode_types,
                                                            db_type=self.db_type,
                                                            load_db_if_exists=self.load_db_if_exists,
                                                            db=self.db,
                                                            n_jobs=self.n_jobs,
                                                            verbose=self.verbose)
        num_docs_before_cut = 0
        use_template = not self.use_openai_model and self.prompt_type not in ['plain'] or self.langchain_only_model
        template, template_if_no_docs, auto_reduce_chunks, query = \
            get_template(query, self.iinput,
                         self.pre_prompt_query, self.prompt_query,
                         self.pre_prompt_summary, self.prompt_summary,
                         self.langchain_action,
                         self.query_action,
                         self.summarize_action,
                         True,  # just to overestimate prompting
                         self.auto_reduce_chunks,
                         self.add_search_to_context,
                         self.system_prompt,
                         self.doc_json_mode)

        # use min_max_new_tokens instead of max_new_tokens for max_new_tokens to get the largest input allowable
        # else max_input_tokens interpreted as user input as smaller than possible and get over-restricted
        max_input_tokens_default = get_max_input_tokens(llm=self.llm, tokenizer=self.tokenizer,
                                                        inference_server=self.inference_server,
                                                        model_name=self.model_name,
                                                        max_new_tokens=self.min_max_new_tokens)
        if self.max_input_tokens >= 0:
            self.max_input_tokens = min(max_input_tokens_default, self.max_input_tokens)
        else:
            self.max_input_tokens = max_input_tokens_default
        model_max_length = get_model_max_length(llm=self.llm, tokenizer=self.tokenizer,
                                                inference_server=self.inference_server,
                                                model_name=self.model_name)

        if hasattr(db, '_persist_directory'):
            lock_file = get_db_lock_file(db, lock_type='sim')
        else:
            base_path = 'locks'
            base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
            name_path = "sim.lock"
            lock_file = os.path.join(base_path, name_path)

        # GET FILTER

        if not is_chroma_db(db):
            # only chroma supports filtering
            chunk_id_filter = None
            filter_kwargs = {}
            filter_kwargs_backup = {}
        else:
            import logging
            logging.getLogger("chromadb").setLevel(logging.ERROR)
            assert self.document_choice is not None, "Document choice was None"
            if isinstance(db, Chroma):
                filter_kwargs_backup = {}  # shouldn't ever need backup
                # chroma >= 0.4
                if len(self.document_choice) == 0 or len(self.document_choice) >= 1 and self.document_choice[
                    0] == DocumentChoice.ALL.value:
                    chunk_id_filter = 0 if self.query_action else -1
                    filter_kwargs = {"filter": {"chunk_id": {"$gte": 0}}} if self.query_action else \
                        {"filter": {"chunk_id": {"$eq": -1}}}
                else:
                    if self.document_choice[0] == DocumentChoice.ALL.value:
                        document_choice = self.document_choice[1:]
                    if len(self.document_choice) == 0:
                        chunk_id_filter = None
                        filter_kwargs = {}
                    elif len(self.document_choice) > 1:
                        chunk_id_filter = None
                        or_filter = [
                            {"$and": [dict(source={"$eq": x}), dict(chunk_id={"$gte": 0})]} if self.query_action else {
                                "$and": [dict(source={"$eq": x}), dict(chunk_id={"$eq": -1})]}
                            for x in self.document_choice]
                        filter_kwargs = dict(filter={"$or": or_filter})
                    else:
                        chunk_id_filter = None
                        # still chromadb UX bug, have to do different thing for 1 vs. 2+ docs when doing filter
                        one_filter = \
                            [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if self.query_action else {
                                "source": {"$eq": x},
                                "chunk_id": {
                                    "$eq": -1}}
                             for x in self.document_choice][0]

                        filter_kwargs = dict(filter={"$and": [dict(source=one_filter['source']),
                                                              dict(chunk_id=one_filter['chunk_id'])]})
            else:
                # migration for chroma < 0.4
                if len(self.document_choice) == 0 or len(self.document_choice) >= 1 and self.document_choice[
                    0] == DocumentChoice.ALL.value:
                    chunk_id_filter = 0 if self.query_action else -1
                    filter_kwargs = {"filter": {"chunk_id": {"$gte": 0}}} if self.query_action else \
                        {"filter": {"chunk_id": {"$eq": -1}}}
                    filter_kwargs_backup = {"filter": {"chunk_id": {"$gte": 0}}}
                elif len(self.document_choice) >= 2:
                    if self.document_choice[0] == DocumentChoice.ALL.value:
                        document_choice = self.document_choice[1:]
                    chunk_id_filter = None
                    or_filter = [
                        {"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if self.query_action else {"source": {"$eq": x},
                                                                                                   "chunk_id": {
                                                                                                       "$eq": -1}}
                        for x in self.document_choice]
                    filter_kwargs = dict(filter={"$or": or_filter})
                    or_filter_backup = [
                        {"source": {"$eq": x}} if self.query_action else {"source": {"$eq": x}}
                        for x in self.document_choice]
                    filter_kwargs_backup = dict(filter={"$or": or_filter_backup})
                elif len(self.document_choice) == 1:
                    chunk_id_filter = None
                    # degenerate UX bug in chroma
                    one_filter = \
                        [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if self.query_action else {
                            "source": {"$eq": x},
                            "chunk_id": {
                                "$eq": -1}}
                         for x in self.document_choice][0]
                    filter_kwargs = dict(filter=one_filter)
                    one_filter_backup = \
                        [{"source": {"$eq": x}} if self.query_action else {"source": {"$eq": x}}
                         for x in self.document_choice][0]
                    filter_kwargs_backup = dict(filter=one_filter_backup)
                else:
                    chunk_id_filter = None
                    # shouldn't reach
                    filter_kwargs = {}
                    filter_kwargs_backup = {}

        # GET DOCS

        if self.document_subset == DocumentSubset.TopKSources.name or query in [None, '', '\n']:
            db_documents, db_metadatas = get_docs_and_meta(db, self.top_k_docs, filter_kwargs=filter_kwargs,
                                                           text_context_list=self.text_context_list,
                                                           chunk_id_filter=chunk_id_filter)
            if len(db_documents) == 0 and filter_kwargs_backup != filter_kwargs:
                db_documents, db_metadatas = get_docs_and_meta(db, self.top_k_docs, filter_kwargs=filter_kwargs_backup,
                                                               text_context_list=self.text_context_list,
                                                               chunk_id_filter=chunk_id_filter)

            if self.top_k_docs == -1:
                top_k_docs = len(db_documents)
            # similar to langchain's chroma's _results_to_docs_and_scores
            docs_with_score = [(Document(page_content=result[0], metadata=result[1] or {}), 0)
                               for result in zip(db_documents, db_metadatas)]
            # remove empty content, e.g. from exception version of document, so don't include empty stuff in
            # summarization
            docs_with_score = [x for x in docs_with_score if x[0].page_content]
            # set in metadata original order of docs
            [x[0].metadata.update(orig_index=ii) for ii, x in enumerate(docs_with_score)]

            # order documents
            doc_hashes = [x.get('doc_hash', 'None') if x.get('doc_hash', 'None') is not None else 'None' for x in
                          db_metadatas]
            if self.query_action:
                doc_chunk_ids = [x.get('chunk_id', 0) if x.get('chunk_id', 0) is not None else 0 for x in db_metadatas]
                docs_with_score2 = [x for hx, cx, x in
                                    sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                                    if cx >= 0]
            else:
                assert self.summarize_action
                doc_chunk_ids = [x.get('chunk_id', -1) if x.get('chunk_id', -1) is not None else -1 for x in
                                 db_metadatas]
                docs_with_score2 = [x for hx, cx, x in
                                    sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                                    if cx == -1
                                    ]
                if len(docs_with_score2) == 0 and len(docs_with_score) > 0:
                    # old database without chunk_id, migration added 0 but didn't make -1 as that would be expensive
                    # just do again and relax filter, let summarize operate on actual chunks if nothing else
                    docs_with_score2 = [x for hx, cx, x in
                                        sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score),
                                               key=lambda x: (x[0], x[1]))
                                        ]
            docs_with_score = docs_with_score2

            docs_with_score = docs_with_score[:self.top_k_docs]
            docs = [x[0] for x in docs_with_score]
            scores = [x[1] for x in docs_with_score]
        else:
            # have query
            # for db=None too
            with filelock.FileLock(lock_file):
                docs_with_score = get_docs_with_score(self.query_embedding, k_db,
                                                      filter_kwargs,
                                                      filter_kwargs_backup,
                                                      db, self.db_type,
                                                      text_context_list=self.text_context_list,
                                                      chunk_id_filter=chunk_id_filter,
                                                      verbose=self.verbose)

        # SELECT PROMPT + DOCS

        self.tokenizer = get_tokenizer(db=db, llm=self.llm, tokenizer=self.tokenizer,
                                       inference_server=self.inference_server,
                                       use_openai_model=self.use_openai_model,
                                       db_type=self.db_type)
        # NOTE: if map_reduce, then no need to auto reduce chunks
        if self.query_action and (self.top_k_docs == -1 or self.auto_reduce_chunks):
            top_k_docs_tokenize = 100
            docs_with_score = docs_with_score[:top_k_docs_tokenize]
            if docs_with_score:
                estimated_prompt_no_docs = template.format(context='', question=query)
            else:
                estimated_prompt_no_docs = template_if_no_docs.format(context='', question=query)
            chat = True  # FIXME?

            # first docs_with_score are most important with highest score
            estimated_full_prompt, \
                query, iinput, context, \
                num_prompt_tokens, max_new_tokens, \
                num_prompt_tokens0, num_prompt_tokens_actual, \
                chat_index, external_handle_chat_conversation, \
                top_k_docs_trial, one_doc_size, \
                truncation_generation = \
                get_limited_prompt(query,
                                   self.iinput,
                                   self.tokenizer,
                                   estimated_instruction=estimated_prompt_no_docs,
                                   prompter=self.prompter,
                                   inference_server=self.inference_server,
                                   prompt_type=self.prompt_type,
                                   prompt_dict=self.prompt_dict,
                                   chat=chat,
                                   max_new_tokens=self.max_new_tokens,
                                   system_prompt=self.system_prompt,
                                   context=self.context,
                                   chat_conversation=self.chat_conversation,
                                   text_context_list=[x[0].page_content for x in docs_with_score],
                                   keep_sources_in_context=self.keep_sources_in_context,
                                   model_max_length=model_max_length,
                                   memory_restriction_level=self.memory_restriction_level,
                                   langchain_mode=self.langchain_mode,
                                   add_chat_history_to_context=self.add_chat_history_to_context,
                                   min_max_new_tokens=self.min_max_new_tokens,
                                   max_input_tokens=self.max_input_tokens,
                                   truncation_generation=self.truncation_generation,
                                   gradio_server=self.gradio_server,
                                   )
            # get updated llm
            self.llm_kwargs.update(max_new_tokens=self.max_new_tokens, context=self.context, iinput=self.iinput)
            if external_handle_chat_conversation:
                # should already have attribute, checking sanity
                assert hasattr(self.llm, 'chat_conversation')
                self.llm_kwargs.update(chat_conversation=self.chat_conversation[chat_index:])
            self.llm, self.model_name, self.streamer, self.prompt_type_out, self.async_output, self.only_new_text, self.gradio_server = \
                get_llm(**self.llm_kwargs)

            # avoid craziness
            if 0 < top_k_docs_trial < self.max_chunks:
                # avoid craziness
                if self.top_k_docs == -1:
                    self.top_k_docs = top_k_docs_trial
                else:
                    self.top_k_docs = min(self.top_k_docs, top_k_docs_trial)
            elif top_k_docs_trial >= self.max_chunks:
                self.top_k_docs = self.max_chunks
            docs_with_score = select_docs_with_score(docs_with_score, self.top_k_docs, one_doc_size)
        else:
            # don't reduce, except listen to top_k_docs and max_total_input_tokens
            one_doc_size = None
            if self.max_total_input_tokens not in [None, -1]:
                # used to limit tokens for summarization, e.g. public instance, over all LLM calls allowed
                self.top_k_docs, one_doc_size, num_doc_tokens = \
                    get_docs_tokens(self.tokenizer,
                                    text_context_list=[x[0].page_content for x in docs_with_score],
                                    max_input_tokens=self.max_total_input_tokens)
            # filter by top_k_docs and maybe one_doc_size
            docs_with_score = select_docs_with_score(docs_with_score, self.top_k_docs, one_doc_size)

        if self.summarize_action:
            # group docs if desired/can to fill context to avoid multiple LLM calls or too large chunks
            docs_with_score, max_doc_tokens = split_merge_docs(docs_with_score,
                                                               self.tokenizer,
                                                               max_input_tokens=self.max_input_tokens,
                                                               docs_token_handling=self.docs_token_handling,
                                                               joiner=self.docs_joiner,
                                                               verbose=self.verbose)
            # in case docs_with_score grew due to splitting, limit again by top_k_docs
            if self.top_k_docs > 0:
                docs_with_score = docs_with_score[:self.top_k_docs]
            # max_input_tokens used min_max_new_tokens as max_new_tokens, so need to assume filled up to that
            # but use actual largest token count
            if '{text}' in template:
                estimated_prompt_no_docs = template.format(text='')
            elif '{input_documents}' in template:
                estimated_prompt_no_docs = template.format(input_documents='')
            elif '{question}' in template:
                estimated_prompt_no_docs = template.format(question=query)
            else:
                estimated_prompt_no_docs = query
            data_point = dict(context=self.context, instruction=estimated_prompt_no_docs or ' ', input=self.iinput)
            prompt_basic = self.prompter.generate_prompt(data_point)
            num_prompt_basic_tokens = get_token_count(prompt_basic, self.tokenizer)

            if self.truncation_generation:
                self.max_new_tokens = model_max_length - max_doc_tokens - num_prompt_basic_tokens
                if os.getenv('HARD_ASSERTS') is not None:
                    # imperfect calculation, so will see how testing does
                    assert self.max_new_tokens >= self.min_max_new_tokens - 50, "%s %s" % (
                        self.max_new_tokens, self.min_max_new_tokens)
            # get updated llm
            self.llm_kwargs.update(max_new_tokens=self.max_new_tokens)
            self.llm, self.model_name, self.streamer, self.prompt_type_out, self.async_output, self.only_new_text, self.gradio_server = \
                get_llm(**self.llm_kwargs)

        # now done with all docs and their sizes, re-order docs if required
        if self.query_action:
            # not relevant for summarization, including in chunk mode, so process docs in order for summarization or
            # extraction put most relevant chunks closest to question, esp. if truncation occurs will be "oldest" or
            # "farthest from response" text that is truncated BUT: for small models, e.g. 6_9 pythia, if it sees some
            # stuff related to h2oGPT first, it can connect that and not listen to rest
            if self.docs_ordering_type in ['best_first']:
                pass
            elif self.docs_ordering_type in ['best_near_prompt', 'reverse_sort']:
                docs_with_score.reverse()
            elif self.docs_ordering_type in ['', None, 'reverse_ucurve_sort']:
                docs_with_score = reverse_ucurve_list(docs_with_score)
            else:
                raise ValueError("No such docs_ordering_type=%s" % self.docs_ordering_type)

        # cut off so no high distance docs/sources considered
        # NOTE: If no query, then distance set was 0 and nothing will be cut
        num_docs_before_cut = len(docs_with_score)
        docs = [x[0] for x in docs_with_score if x[1] < self.cut_distance]
        scores = [x[1] for x in docs_with_score if x[1] < self.cut_distance]
        if len(scores) > 0 and self.verbose:
            print("Distance: min: %s max: %s mean: %s median: %s" %
                  (scores[0], scores[-1], np.mean(scores), np.median(scores)), flush=True)

        # if HF type and have no docs, could bail out, but makes code too complex

        if self.document_subset in non_query_commands:
            # no LLM use at all, just sources
            return docs, None, [], num_docs_before_cut, self.use_llm_if_no_docs, self.top_k_docs_max_show, \
                self.llm, self.model_name, self.streamer, self.prompt_type_out, self.async_output, self.only_new_text

        # FIXME: WIP
        common_words_file = "data/NGSL_1.2_stats.csv.zip"
        if False and os.path.isfile(common_words_file) and self.langchain_action == LangChainAction.QUERY.value:
            df = pd.read_csv("data/NGSL_1.2_stats.csv.zip")
            import string
            reduced_query = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).strip()
            reduced_query_words = reduced_query.split(' ')
            set_common = set(df['Lemma'].values.tolist())
            num_common = len([x.lower() in set_common for x in reduced_query_words])
            frac_common = num_common / len(reduced_query) if reduced_query else 0
            # FIXME: report to user bad query that uses too many common words
            if self.verbose:
                print("frac_common: %s" % frac_common, flush=True)

        if len(docs) == 0:
            # avoid context == in prompt then
            template = template_if_no_docs

        got_any_docs = len(docs) > 0
        # update template in case situation changed or did get docs
        # then no new documents from database or not used, redo template
        # got template earlier as estimate of template token size, here is final used version
        template, template_if_no_docs, self.auto_reduce_chunks, query = \
            get_template(query, self.iinput,
                         self.pre_prompt_query, self.prompt_query,
                         self.pre_prompt_summary, self.prompt_summary,
                         self.langchain_action,
                         self.query_action,
                         self.summarize_action,
                         got_any_docs,
                         self.auto_reduce_chunks,
                         self.add_search_to_context,
                         self.system_prompt,
                         self.doc_json_mode)

        if self.doc_json_mode:
            # make copy so don't change originals
            docs = [Document(page_content=json.dumps(dict(ID=xi, content=x.page_content)),
                             metadata=copy.deepcopy(x.metadata) or {})
                    for xi, x in enumerate(docs)]

        if self.langchain_action == LangChainAction.QUERY.value:
            if use_template:
                # instruct-like, rather than few-shot prompt_type='plain' as default
                # but then sources confuse the model with how inserted among rest of text, so avoid
                prompt = PromptTemplate(
                    # input_variables=["summaries", "question"],
                    input_variables=["context", "question"],
                    template=template,
                )
                chain = load_qa_chain(self.llm, prompt=prompt, verbose=self.verbose)
            else:
                # only if use_openai_model = True, unused normally except in testing
                chain = load_qa_with_sources_chain(self.llm)
            chain_kwargs = dict(input_documents=docs, question=query)
            target = wrapped_partial(chain, chain_kwargs)
        elif self.summarize_action:
            if self.async_output:
                return_intermediate_steps = False
            else:
                return_intermediate_steps = True
            if self.langchain_action == LangChainAction.SUMMARIZE_MAP.value:
                prompt = PromptTemplate(input_variables=["text"], template=template)
                # token_max is per llm call
                chain = load_general_summarization_chain(self.llm, chain_type="map_reduce",
                                                         map_prompt=prompt, combine_prompt=prompt,
                                                         return_intermediate_steps=return_intermediate_steps,
                                                         token_max=self.max_input_tokens, verbose=self.verbose)
                if self.async_output:
                    chain_func = chain.arun
                else:
                    chain_func = chain
                target = wrapped_partial(chain_func, dict(input_documents=docs,
                                                          token_max=self.max_input_tokens))  # , return_only_outputs
                # =True)
            elif self.langchain_action == LangChainAction.SUMMARIZE_ALL.value:
                assert use_template
                prompt = PromptTemplate(input_variables=["text"], template=template)
                chain = load_general_summarization_chain(self.llm, chain_type="stuff", prompt=prompt,
                                                         return_intermediate_steps=return_intermediate_steps,
                                                         verbose=self.verbose)
                if self.async_output:
                    chain_func = chain.arun
                else:
                    chain_func = chain
                target = wrapped_partial(chain_func)
            elif self.langchain_action == LangChainAction.SUMMARIZE_REFINE.value:
                chain = load_general_summarization_chain(self.llm, chain_type="refine",
                                                         return_intermediate_steps=return_intermediate_steps,
                                                         verbose=self.verbose)
                if self.async_output:
                    chain_func = chain.arun
                else:
                    chain_func = chain
                target = wrapped_partial(chain_func)
            elif self.langchain_action == LangChainAction.EXTRACT.value:
                prompt = PromptTemplate(input_variables=["text"], template=template)
                chain = load_general_summarization_chain(self.llm, chain_type="map",
                                                         map_prompt=prompt, combine_prompt=prompt,
                                                         return_intermediate_steps=return_intermediate_steps,
                                                         token_max=self.max_input_tokens, verbose=self.verbose)
                if self.async_output:
                    chain_func = chain.arun
                else:
                    chain_func = chain
                target = wrapped_partial(chain_func, dict(input_documents=docs,
                                                          token_max=self.max_input_tokens))  # , return_only_outputs
                # =True)
            else:
                raise RuntimeError("No such langchain_action=%s" % self.langchain_action)
        else:
            raise RuntimeError("No such langchain_action=%s" % self.langchain_action)

        return docs, target, scores, num_docs_before_cut, self.use_llm_if_no_docs, self.top_k_docs_max_show, \
            self.llm, self.model_name, self.streamer, self.prompt_type_out, self.async_output, self.only_new_text
