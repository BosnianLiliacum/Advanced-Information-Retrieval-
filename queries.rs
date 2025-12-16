
// DEFAULT CODE
// use helix_db::helix_engine::traversal_core::config::Config;

// pub fn config() -> Option<Config> {
//     None
// }



use bumpalo::Bump;
use heed3::RoTxn;
use helix_macros::{handler, tool_call, mcp_handler, migration};
use helix_db::{
    helix_engine::{
        reranker::{
            RerankAdapter,
            fusion::{RRFReranker, MMRReranker, DistanceMethod},
        },
        traversal_core::{
            config::{Config, GraphConfig, VectorConfig},
            ops::{
                bm25::search_bm25::SearchBM25Adapter,
                g::G,
                in_::{in_::InAdapter, in_e::InEdgesAdapter, to_n::ToNAdapter, to_v::ToVAdapter},
                out::{
                    from_n::FromNAdapter, from_v::FromVAdapter, out::OutAdapter, out_e::OutEdgesAdapter,
                },
                source::{
                    add_e::AddEAdapter,
                    add_n::AddNAdapter,
                    e_from_id::EFromIdAdapter,
                    e_from_type::EFromTypeAdapter,
                    n_from_id::NFromIdAdapter,
                    n_from_index::NFromIndexAdapter,
                    n_from_type::NFromTypeAdapter,
                    v_from_id::VFromIdAdapter,
                    v_from_type::VFromTypeAdapter
                },
                util::{
                    dedup::DedupAdapter, drop::Drop, exist::Exist, filter_mut::FilterMut,
                    filter_ref::FilterRefAdapter, map::MapAdapter, paths::{PathAlgorithm, ShortestPathAdapter},
                    range::RangeAdapter, update::UpdateAdapter, order::OrderByAdapter,
                    aggregate::AggregateAdapter, group_by::GroupByAdapter, count::CountAdapter,
                },
                vectors::{
                    brute_force_search::BruteForceSearchVAdapter, insert::InsertVAdapter,
                    search::SearchVAdapter,
                },
            },
            traversal_value::TraversalValue,
        },
        types::GraphError,
        vector_core::vector::HVector,
    },
    helix_gateway::{
        embedding_providers::{EmbeddingModel, get_embedding_model},
        router::router::{HandlerInput, IoContFn},
        mcp::mcp::{MCPHandlerSubmission, MCPToolInput, MCPHandler}
    },
    node_matches, props, embed, embed_async,
    field_addition_from_old_field, field_type_cast, field_addition_from_value,
    protocol::{
        response::Response,
        value::{casting::{cast, CastType}, Value},
        format::Format,
    },
    utils::{
        id::{ID, uuid_str},
        items::{Edge, Node},
        properties::ImmutablePropertiesMap,
    },
};
use sonic_rs::{Deserialize, Serialize, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use chrono::{DateTime, Utc};

// Re-export scalar types for generated code
type I8 = i8;
type I16 = i16;
type I32 = i32;
type I64 = i64;
type U8 = u8;
type U16 = u16;
type U32 = u32;
type U64 = u64;
type U128 = u128;
type F32 = f32;
type F64 = f64;
    
pub fn config() -> Option<Config> {
return Some(Config {
vector_config: Some(VectorConfig {
m: Some(16),
ef_construction: Some(128),
ef_search: Some(768),
}),
graph_config: Some(GraphConfig {
secondary_indices: None,
}),
db_max_size_gb: Some(10),
mcp: Some(true),
bm25: Some(true),
schema: Some(r#"{
  "schema": {
    "nodes": [
      {
        "name": "Comment",
        "properties": {
          "content": "String",
          "score": "I32",
          "id": "ID",
          "label": "String"
        }
      },
      {
        "name": "Post",
        "properties": {
          "content": "String",
          "url": "String",
          "id": "ID",
          "subreddit": "String",
          "score": "I32",
          "title": "String",
          "label": "String"
        }
      }
    ],
    "vectors": [
      {
        "name": "Content",
        "properties": {
          "label": "String",
          "data": "Array(F64)",
          "id": "ID",
          "chunk": "String",
          "score": "F64"
        }
      }
    ],
    "edges": [
      {
        "name": "CommentOf",
        "from": "Post",
        "to": "Comment",
        "properties": {}
      },
      {
        "name": "EmbeddingOf",
        "from": "Post",
        "to": "Content",
        "properties": {}
      }
    ]
  },
  "queries": [
    {
      "name": "load_a_post",
      "parameters": {
        "subreddit": "String",
        "vector": "Array(F64)",
        "score": "I32",
        "comments": "Array({ic_score: I32ic_content: String})",
        "url": "String",
        "title": "String",
        "content": "String"
      },
      "returns": []
    },
    {
      "name": "get_all_posts",
      "parameters": {},
      "returns": [
        "posts"
      ]
    },
    {
      "name": "search_posts_vec",
      "parameters": {
        "query": "Array(F64)",
        "k": "I32"
      },
      "returns": []
    }
  ]
}"#.to_string()),
embedding_model: Some("text-embedding-ada-002".to_string()),
graphvis_node_label: None,
})
}
pub struct Post {
    pub subreddit: String,
    pub title: String,
    pub content: String,
    pub url: String,
    pub score: i32,
}

pub struct Comment {
    pub content: String,
    pub score: i32,
}

pub struct EmbeddingOf {
    pub from: Post,
    pub to: Content,
}

pub struct CommentOf {
    pub from: Post,
    pub to: Comment,
}

pub struct Content {
    pub chunk: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct load_a_postInput {

pub subreddit: String,
pub title: String,
pub content: String,
pub vector: Vec<f64>,
pub url: String,
pub score: i32,
pub comments: Vec<commentsData>
}
#[derive(Serialize, Deserialize, Clone)]
pub struct commentsData {
    pub ic_score: i32,
    pub ic_content: String,
}
#[handler]
pub fn load_a_post (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<load_a_postInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let post_node = G::new_mut(&db, &arena, &mut txn)
.add_n("Post", Some(ImmutablePropertiesMap::new(5, vec![("content", Value::from(&data.content)), ("score", Value::from(&data.score)), ("title", Value::from(&data.title)), ("subreddit", Value::from(&data.subreddit)), ("url", Value::from(&data.url))].into_iter(), &arena)), None).collect_to_obj()?;
    let vec = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&i_vector, "Content", Some(ImmutablePropertiesMap::new(0, vec![].into_iter(), &arena))).collect_to_obj()?;
    G::new_mut(&db, &arena, &mut txn)
.add_edge("EmbeddingOf", None, post_node.id(), vec.id(), false).collect_to_obj()?;
    for commentsData { ic_content, ic_score } in &data.comments {
    let comment_node = G::new_mut(&db, &arena, &mut txn)
.add_n("Comment", Some(ImmutablePropertiesMap::new(2, vec![("score", Value::from(&ic_score)), ("content", Value::from(&ic_content))].into_iter(), &arena)), None).collect_to_obj()?;
    G::new_mut(&db, &arena, &mut txn)
.add_edge("CommentOf", None, post_node.id(), comment_node.id(), false).collect_to_obj()?;
}
;
let response = json!({
    "data": "success"
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize)]
pub struct Get_all_postsPostsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub url: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub subreddit: Option<&'a Value>,
    pub title: Option<&'a Value>,
}

#[handler]
pub fn get_all_posts (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let posts = G::new(&db, &txn, &arena)
.n_from_type("Post").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "posts": posts.iter().map(|post| Get_all_postsPostsReturnType {
        id: uuid_str(post.id(), &arena),
        label: post.label(),
        url: post.get_property("url"),
        content: post.get_property("content"),
        subreddit: post.get_property("subreddit"),
        title: post.get_property("title"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_posts_vecInput {

pub query: Vec<f64>,
pub k: i32
}
#[derive(Serialize)]
pub struct Search_posts_vecPostsReturnType<'a> {
    pub subreddit: Option<&'a Value>,
    pub title: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub url: Option<&'a Value>,
}

#[handler]
pub fn search_posts_vec (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_posts_vecInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let vecs = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.query, data.k.clone(), "Content", None).collect::<Result<Vec<_>, _>>()?;
    let posts = G::from_iter(&db, &txn, vecs.iter().cloned(), &arena)

.in_node("EmbeddingOf").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "posts": posts.iter().map(|post| Search_posts_vecPostsReturnType {
        subreddit: post.get_property("subreddit"),
        title: post.get_property("title"),
        content: post.get_property("content"),
        url: post.get_property("url"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}


