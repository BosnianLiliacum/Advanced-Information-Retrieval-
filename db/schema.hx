N::Post {
    subreddit: String,
    title: String,
    content: String,
    url: String,
    score: I32,
    comments: [String],
}

V::Content {
    content: [F64]
}

E::EmbeddingOf {
    From: Post,
    To: Content,
}
