# AGENTS.md

## 目的

这份文件是给后续编码 agent 的仓库速查手册。目标是先帮助你快速定位代码，再提醒你哪些地方改动需要联动、哪些命令最适合验证。

## 仓库定位

- 这是一个前后端同仓库的 Open WebUI 项目。
- 前端是 `Svelte 5 + SvelteKit + Vite + Tailwind 4`。
- 后端是 `FastAPI + Socket.IO + SQLAlchemy`，并同时保留了 `Alembic` 和历史 `Peewee` 迁移链。
- 生产模式下，前端会构建到根目录 `build/`，随后由后端在运行时作为 SPA 静态资源提供。
- 根目录 `package.json` 是版本号来源；`pyproject.toml` 通过 Hatch 从它读取版本。

## 先看哪里

- 前端应用入口：`src/routes/+layout.svelte`
- 主聊天页：`src/routes/(app)/+page.svelte`
- 前端 API 封装：`src/lib/apis/*`
- 前端全局状态：`src/lib/stores/index.ts`
- 后端入口：`backend/open_webui/main.py`
- 运行时配置：`backend/open_webui/env.py`、`backend/open_webui/config.py`
- 数据库与会话：`backend/open_webui/internal/db.py`
- API 路由：`backend/open_webui/routers/*.py`
- 数据模型与表单/查询辅助：`backend/open_webui/models/*.py`
- 领域工具逻辑：`backend/open_webui/utils/*.py`
- WebSocket：`backend/open_webui/socket/*`
- 当前 Alembic 迁移：`backend/open_webui/migrations/versions/*`
- 历史 Peewee 迁移：`backend/open_webui/internal/migrations/*`
- E2E：`cypress/`
- Python 测试：`backend/open_webui/test/`
- Pyodide 资源准备脚本：`scripts/prepare-pyodide.js`

## 代码组织规律

- 前端和后端大多按领域一一对应。
- 常见链路是：`src/routes` -> `src/lib/components` -> `src/lib/apis/<domain>` -> `backend/open_webui/routers/<domain>.py` -> `backend/open_webui/models/<domain>.py` 与 `backend/open_webui/utils/*.py`
- `backend/open_webui/models/*.py` 不是“纯 ORM 层”，通常同时放：
  - SQLAlchemy 模型
  - Pydantic 表单/响应模型
  - 查询和写入辅助方法
- `backend/open_webui/config.py` 中很多配置不是单纯环境变量，而是通过 `PersistentConfig` 持久化到数据库；改配置相关逻辑时不要默认它只活在 `.env`。
- `src/routes/+layout.svelte` 负责会话初始化、i18n、socket 连接、Pyodide worker 和全局 UI 外壳；很多“看起来像局部页面问题”的行为，实际上由这里驱动。

## 主要功能入口

- 登录、用户与权限：
  - 前端：`src/routes/auth/*`、`src/lib/apis/auths`、`src/lib/apis/users`
  - 后端：`backend/open_webui/routers/auths.py`、`backend/open_webui/routers/users.py`、`backend/open_webui/utils/auth.py`
- 聊天与消息：
  - 前端：`src/routes/(app)/+page.svelte`、`src/lib/components/chat/*`、`src/lib/apis/chats`、`src/lib/apis/openai`
  - 后端：`backend/open_webui/routers/chats.py`、`backend/open_webui/routers/openai.py`、`backend/open_webui/models/chats.py`、`backend/open_webui/socket/main.py`
- 频道：
  - 前端：`src/routes/(app)/channels/[id]/+page.svelte`、`src/lib/components/channel/*`
  - 后端：`backend/open_webui/routers/channels.py`、`backend/open_webui/models/channels.py`
- 工作区与管理后台：
  - 前端：`src/routes/(app)/workspace/*`、`src/routes/(app)/admin/*`
  - 后端：对应的 `models.py`、`knowledge.py`、`prompts.py`、`tools.py`、`skills.py`、`functions.py`、`groups.py`、`analytics.py`、`evaluations.py`
- RAG / 检索 / 文档：
  - 前端：`src/routes/(app)/workspace/knowledge/*`、`src/lib/apis/retrieval`、`src/lib/apis/knowledge`
  - 后端：`backend/open_webui/routers/retrieval.py`、`backend/open_webui/routers/knowledge.py`、`backend/open_webui/utils/embeddings.py`、`backend/open_webui/retrieval/*`

## 本地开发命令

- 安装前端依赖：`npm ci --force`
- 启动前端开发服务器：`npm run dev`
- 使用 5050 端口启动前端：`npm run dev:5050`
- 启动后端开发服务器：`cd backend && ./dev.sh`
- 直接启动后端：`cd backend && uvicorn open_webui.main:app --host 0.0.0.0 --port 8080 --forwarded-allow-ips '*' --reload`
- Docker Compose 启动：`docker compose up -d --build`
- 前端类型检查：`npm run check`
- 前端单测：`npm run test:frontend`
- 前端构建：`npm run build`
- 前端格式化：`npm run format`
- 后端格式化：`npm run format:backend`
- 语言包提取：`npm run i18n:parse`
- Python 测试：`pytest backend/open_webui/test -q`

## 运行与环境约定

- 开发模式下，前端通过 `src/lib/constants.ts` 把 API 默认指向 `http://<当前主机>:8080`。
- 这意味着 `npm run dev` 只会启动 Vite，不会自动启动后端；本地联调时通常要同时开一个 `backend/dev.sh`。
- 默认 Docker Compose 把容器内 `8080` 映射到宿主机 `3000`。
- `backend/start.sh` 在缺少密钥时会自动生成并读取 `.webui_secret_key`。
- `npm run dev` 和 `npm run build` 都会先执行 `scripts/prepare-pyodide.js`。这个脚本会复制并下载 Pyodide 相关资源，所以在网络受限环境里可能变慢或失败。

## 修改时的联动原则

- 改 API 合约时，通常要同步更新：
  - `backend/open_webui/routers/<domain>.py`
  - `backend/open_webui/models/<domain>.py`
  - `src/lib/apis/<domain>/index.ts`
  - 受影响的 Svelte 页面、组件和 store
- 改聊天、流式响应、在线状态或协作相关行为时，务必检查 `backend/open_webui/socket/main.py` 和 `src/routes/+layout.svelte`。
- 改数据库 schema 时：
  - 优先新增 Alembic revision 到 `backend/open_webui/migrations/versions/`
  - 不要直接改历史迁移文件
  - 同时确认 `backend/open_webui/internal/migrations/` 的旧启动链是否会受影响，因为它仍会在启动时执行
- 改 Python 依赖时，至少检查：
  - `pyproject.toml`
  - `backend/requirements.txt`
  - 如果该依赖对最小安装也必要，再检查 `backend/requirements-min.txt`
- 改版本号时，先改 `package.json`；Python 包版本会从这里读取。
- 改前端静态资源或构建链时，顺手确认：
  - `vite.config.ts`
  - `svelte.config.js`
  - `Dockerfile`

## 验证建议

- 只改 Svelte 视图或状态：
  - 至少跑 `npm run check`
- 改前端 API 交互或业务逻辑：
  - 建议跑 `npm run check`
  - 如涉及可测试逻辑，再跑 `npm run test:frontend`
  - 如改构建、路由或静态资源，再跑 `npm run build`
- 改后端业务逻辑：
  - 优先跑受影响的 `pytest` 或最接近的目标测试
  - 当前仓库内 Python 测试覆盖较少，现有测试主要集中在 `backend/open_webui/test/util/test_redis.py`
- 改配置、启动流程、迁移、数据库连接或导入路径：
  - 最好实际启动一次后端做 smoke test
  - 启动相关问题经常会在模块 import、迁移执行或配置加载阶段暴露
- 改用户主流程：
  - 视范围考虑补跑 Cypress 相关用例

## CI 与提交前检查

- 前端 CI 会做格式化、`i18n:parse`、构建和 `vitest`。
- 后端 CI 会检查 `black` 格式化结果。
- 迁移相关 CI 会分别用 SQLite 和 Postgres 拉起后端做启动验证。
- 如果你新增了文案 key，别忘了 `npm run i18n:parse` 可能会产生额外变更。

## 已知坑点

- `backend/open_webui/main.py` 很大，不适合从头通读；优先用 `include_router`、`mount`、路由前缀和目标函数名做定点阅读。
- `src/routes/(app)/home/+page.svelte` 当前是空文件，不要把它误判成缺失实现；实际主聊天页在 `src/routes/(app)/+page.svelte`。
- 前端根布局里已经有大量全局初始化逻辑；如果页面上出现“刷新后才正常”或“只在首次载入异常”的问题，先回头看 `src/routes/+layout.svelte`。
- 当前后端 API 增长较大时，优先沿用既有 domain 文件，而不是新建一层抽象把现有模式打散。

## 推荐工作方式

- 先锁定功能所在领域，再沿“页面/组件 -> API 封装 -> router -> model/utils”的链路追踪。
- 优先做小而完整的改动，避免跨多个领域同时重构。
- 提交前说明你实际跑过哪些命令，哪些因为时间、依赖或环境原因没跑。
