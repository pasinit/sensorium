"""sensorium: multisensory sensorimotor episodes for a body-agnostic world model.

Modules:
    episode     The one episode format every body produces (save / load / validate).
    audio       Synthetic audio from simulator state + log-mel features.
    describe    Templated state descriptions (language-probe labels only).
    bodies      One adapter per body: car, drone, piano.
    generate    CLI: config file -> episodes + manifest.
    browse      CLI: episodes -> static HTML pages for human review.
"""
