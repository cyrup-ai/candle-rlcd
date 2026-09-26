# CPU image of candle-rlcd. Serves convaiinnovations/laya by default; the weights download to
# the /data volume on first start, so mount it to keep them between runs:
#
#   docker run -p 8080:8080 -v candle-rlcd:/data ghcr.io/cyrup-ai/candle-rlcd
#   docker run -p 8080:8080 -v candle-rlcd:/data ghcr.io/cyrup-ai/candle-rlcd \
#       serve --host 0.0.0.0 --model convaiinnovations/laya/multilingual

FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
RUN cargo build --release --locked --bin candle-rlcd

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/candle-rlcd /usr/local/bin/candle-rlcd
ENV HF_HOME=/data/huggingface
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["candle-rlcd"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
