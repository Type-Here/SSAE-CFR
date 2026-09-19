"""Expert-prior construction: a structured domain prior authored by a frozen LLM.

A second, independent prior artifact beside the SVD prior of `prior_build`. Where
that one embeds a one-line gloss per covariate and factors the result, this one asks
the same frozen model to write a structured knowledge document - one FeatureCard per
covariate plus a small ConceptBank - and embeds the paragraphs it wrote.

The two never share an artifact directory and never overwrite each other. Nothing in
`model_v3`, the adapters, training, losses or evaluation reads this package yet; it
builds the artifact and stops there.

Pipeline, one stage per module, in order:

    inputs    load and validate the three researcher-authored YAMLs, separately
    prompt    compose them into one prompt, deterministically, and hash it
    generate  one call to the frozen model
    schema    parse and validate the response against the contract, repairing nothing
    cards     collect the expert's own embedding_text, verbatim
    embed     one documented pooling strategy over the same frozen model
    build     the CLI that runs the stages and writes the manifest

The contract is `semantic_inputs/<dataset>/<dataset>_expert_prior_request.yaml`. The
validator is generated from it at runtime and keeps no second copy of it in code.
"""
