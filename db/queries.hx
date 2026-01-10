QUERY upload_a_post(
    subreddit: String,
    title: String,
    content: String,
    vector: [F64],
    url: String,
    score: I32,
    comments: [String]
) =>
    post_node <- AddN<Post>({
        subreddit: subreddit,
        title: title,
        content: content,
        url: url,
        score: score,
        comments: comments,
    })

    vec <- AddV<Content>(vector)
    AddE<EmbeddingOf>::From(post_node)::To(vec)

    RETURN "success"

QUERY get_all_posts() =>
    posts <- N<Post>
    RETURN posts

QUERY search_posts_vec(query: [F64], k: I32) =>
    vecs <- SearchV<Content>(query, k)
    posts <- vecs::In<EmbeddingOf>
    RETURN posts
