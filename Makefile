
tests: FORSE
	pyclean .; \
	PYTHONPATH=src pytest -vv src


codestyle: FORSE
	isort --line-length=120 --profile=black src && \
	black --line-length=120 src

codestyle-check: FORSE
	isort --line-length=120 --profile=black --check-only src && \
	black --line-length=120 --check src && \
	flake8 --max-line-length 120 --ignore=Q000,D100,D205,D212,D400,D415,W605 src

pod:
	apt install zsh && \
	sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
	cd ~ && git clone https://github.com/broniyasinnik/dotfiles.git && \
	mv dotfiles/* . && mv dotfiles/.* . && rm -r dotfiles && \
	apt-get install autojump && \
	omz reload && \


environment: environment.yml
	conda env create -f environment.yml && \
	exec zsh && \
	conda init zsh && omz reload && \
	conda activate churn && \
	python -m ipykernel install --user --name=churn

tensorboard:
	tensorboard --logdir ./assets/logs --port 6001

train_small:
	python -m src.trainer -cn experiment_small \
 			components.macro_features=True \
 			components.activity_seq=True \
 			components.action_graphs=True

train:
	python -m src.trainer -cn experiment \
			experiment_name="gnn_pattern_traj" \
 			components.macro_features.include=True \
 			components.activity_seq.include=True \
 			components.action_graphs.include=True \
 			components.patterns_seq.include=True\
 			components.action_graphs.node_label='event_id' \
 			model.gnn.gcn_in=92\
 			batch_size=128


FORSE: