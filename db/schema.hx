N::Post {
    subreddit: String,
    title: String,
    content: String,
    url: String,
    score: I32,
}

V::Content {
    content: [F64]
}

E::EmbeddingOf {
    From: Post,
    To: Content,
}

N::Comment {
    content: String,
    score: I32,
}

E::CommentOf {
    From: Post,
    To: Comment,
}
