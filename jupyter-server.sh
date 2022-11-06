#!/bin/bash
docker run -it --rm -v /home/waynechen/Project/Github/wenwen357951/Intracranial-Hemorrhage:/ich -p 8080:8888 -d wenwen357951/intracerebral-haemorrhage:latest-jupyter
