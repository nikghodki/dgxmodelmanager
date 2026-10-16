# Convenience wrappers; ./install.sh --help documents the real interface.
SHELL := /bin/bash
.PHONY: install upgrade status logs uninstall dry-run test lint

install:            ## install or upgrade in place
	./install.sh install

upgrade:            ## git pull --ff-only, then install
	./install.sh upgrade

status:
	./install.sh status

logs:               ## follow the dgxmm container log
	./install.sh logs -f

uninstall:          ## remove container + image (keeps data, secrets, config)
	./install.sh uninstall

dry-run:
	./install.sh --dry-run

test:               ## backend + installer tests
	.venv/bin/pytest -q

lint:               ## shellcheck the installer
	docker run --rm -v "$(CURDIR):/mnt" -w /mnt koalaman/shellcheck:stable install.sh tests/fixtures/install_stubs/*
