import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    alpha = 0.75
    beta = 0.15
    num_docs = vec_docs.shape[0]
    num_queries = vec_queries.shape[0]
    num_its = 3

    for its in range(num_its):
        for i in range(num_queries):
            rel = np.argsort(-sim[:, i])[:n]
            A = alpha*np.sum(vec_docs[np.array(rel), :], axis = 0) - beta*np.sum(vec_docs[~np.array(rel), :], axis = 0)
            vec_queries[i,:] += A
        sim = cosine_similarity(vec_docs, vec_queries)        

    return sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    alpha = 0.8
    beta = 0.2
    num_docs = vec_docs.shape[0]
    num_queries = vec_queries.shape[0]
    num_its = 3
    nn = 10
    names = tfidf_model.get_feature_names()

    for its in range(num_its):
        for i in range(num_queries):
            rel = np.argsort(-sim[:, i])[:n]
            rel_docs = vec_docs[np.array(rel), :]
            A = alpha*np.sum(vec_docs[np.array(rel), :], axis = 0) - beta*np.sum(vec_docs[~np.array(rel), :], axis = 0)
            vec_queries[i,:] += A
            # query expansion
            ll = []
            for row in rel:
                ele = vec_docs.getrow(row).toarray()[0].ravel()
                top10 = ele.argsort()[-nn:]
                ss = ''
                for k in top10:
                    ss += names[k] + ' '
                ll.append(ss)
            # np.sum(tfidf_model.transform(ll), axis = 0)
            vec_queries[i, :] += np.sum(tfidf_model.transform(ll), axis = 0)
           
        sim = cosine_similarity(vec_docs, vec_queries) 

    return sim