// Start writing your queries here.
//
// You can use the schema to help you write your queries.
//
// Queries take the form:
//     QUERY {query name}({input name}: {input type}) =>
//         {variable} <- {traversal}
//         RETURN {variable}
//
// Example:
//     QUERY GetUserFriends(user_id: String) =>
//         friends <- N<User>(user_id)::Out<Knows>
//         RETURN friends
//
//
// For more information on how to write queries,
// see the documentation at https://docs.helix-db.com
// or checkout our GitHub at https://github.com/HelixDB/helix-db

QUERY load_a_post(
    subreddit: String,
    title: String,
    content: String,
    vector: [F64],
    url: String,
    score: I32,
    comments: [{ic_content: String, ic_score: I32}]
) =>
    post_node <- AddN<Post>({
        subreddit: subreddit,
        title: title,
        content: content,
        url: url,
        score: score
    })

    vec <- AddV<Content>(i_vector)
    AddE<EmbeddingOf>::From(post_node)::To(vec)

    FOR {ic_content, ic_score} IN comments {
        comment_node <- AddN<Comment>({
            content: ic_content,
            score: ic_score
        })
        AddE<CommentOf>::From(post_node)::To(comment_node)
    }

    RETURN "success"

QUERY get_all_posts() =>
    posts <- N<Post>
    RETURN posts

QUERY search_posts_vec(query: [F64], k: I32) =>
    vecs <- SearchV<Content>(query, k)
    posts <- vecs::In<EmbeddingOf>
    RETURN posts::{subreddit, title, content, url}

//QUERY search_posts_vec_w_comments(query: [F64], k: I32) =>
//    vecs <- SearchV<Embedding>(query, k)
//    posts <- vecs::In<EmbeddingOf>
//    RETURN posts::{subreddit, title, content, url}
