# Database Structure and Operations

## Overview

The system uses two types of databases:
1. Personal Database: Stores memories specific to each container.
2. Master Database: Stores memories from all containers.

## Database Structure

Both personal and master databases have the following table structure:

### Memories Table

| Column    | Type    | Description                            |
|-----------|---------|----------------------------------------|
| id        | INTEGER | Primary key                            |
| query     | TEXT    | The original query                     |
| result    | TEXT    | The response or result                 |
| embedding | BLOB    | Vector representation of the query     |
| timestamp | DATETIME| When the memory was created/updated    |
| author    | TEXT    | (Master DB only) Container ID          |

## Querying Process

1. When a query is received, it's first converted to an embedding.
2. The system searches the personal database for similar memories.
3. If fewer than 3 relevant memories are found, it also searches the master database.
4. Memories are ranked by similarity and the top results are returned.

## Syncing Mechanism

- New memories are saved to both the personal and master databases.
- The master database includes an 'author' field to track which container created each memory.
- There's no automatic sync between personal and master databases after initial creation.

## Versioning and Change Tracking

(To be implemented)