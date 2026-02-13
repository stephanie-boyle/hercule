# HERCULE - Healthcare Epidemiology Reasoning through Cross-domain Unified Link Embeddings

In collaboration with Charlotte Ristic for our Language Models for Structured Data course. 

This is a work in progress! 

## Problem statement:
Integrating epidemiological surveillance with pharmaceutical logistics is a critical yet challenging task in global health informatics with broad real-world impact. While recent advances in data collection have enabled significant progress in tracking disease outbreaks, current solutions remain limited by a disconnect between surveillance signals and logistical action. We introduce HERCULE (Healthcare Epidemiology Reasoning through Cross-domain Unified Link Embeddings), a pipeline for logistical inference using Knowledge Graph Embeddings (KGE) and semantic map- ping. Designed for accuracy and semantic interoperability, HERCULE empowers health organizations by providing access to automated transitive inference over a multimodal biomedical knowledge graph. 

## How to navigate:
In source you'll see that the pipeline consists of:
1. extraction
2. graph
3. visualisation / storage

There is a separate file called main, this is how the pipeline is run.


needs updating.


## TODO: 
1. Add country 3 letter code to name mapping
2. Change data extraction so it only runs if the files don't already exist for the day.
3. Review the use of country outbreak reports instead - there is a great one from the brown pandemic centre.
4. Look at the disease codes to check there correct and consider standardising codes.
5. Check use of neo4j instead of static graphs.