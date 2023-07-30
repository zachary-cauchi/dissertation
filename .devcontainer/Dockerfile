FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

ARG USERNAME=snmn
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USER_UID:$USER_GID /workspace

RUN mkdir -p /workspaces/snmn/.vscode/stubs \
    && ln -s /usr/local/lib/python3.8/dist-packages/tensorflow_core/ /workspaces/snmn/.vscode/stubs/tensorflow

USER ${USERNAME}

COPY --chown=${USERNAME}:${USERNAME} requirements.txt /tmp/
RUN pip install --user -r /tmp/requirements.txt