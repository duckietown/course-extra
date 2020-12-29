#  Lisus: Assignment 3 {#lisus-assignment3 status=ready}

## The Final Result {#lisus-assignment3-final-result}

The goal of this assignment was to train a Gaussian Process (GP) model to predict the state of the lane that the Duckiebot is currently in. The idea was to take the detected white and yellow line segments and to use them as inputs to predict whether the current lane is in a "straight", "left", or "right" portion. 

The final results of the GP perfomance is shown below. The visualization shows the detected segments with an x marking the average position of each type of segments. The true and predicted lane state is listed in the title. Please refer to [instructions to reproduce](#instructions-lisus-assignment3) to reproduce these results.

<figure class="flow-subfigures">       
  <figcaption>Examples of succesfully predicted states. Please refer to [instructions to reproduce](#instructions-lisus-assignment3) to reproduce these results.</figcaption>     
  <figure>         
    <figcaption>Left state</figcaption>         
    <img style='width:30em' src="corr_left.png"/>     
  </figure>     
  <figure>           
    <figcaption>Straight state</figcaption>         
    <img style='width:30em' src="corr_straight.png"/>     
  </figure> 
  <figure>           
    <figcaption>Right state</figcaption>         
    <img style='width:30em' src="corr_right.png"/>     
  </figure> 
</figure>



## Mission and Scope {#lisus-assignment3-final-scope}

The 


### Motivation {#lisus-assignment3-final-result-motivation}

_Now step back and tell us how you got to that mission._

- What are we talking about? [Brief introduction / problem in general terms]

- Why is it important? [Relevance]

### Existing solution {#lisus-assignment3-final-literature}

-  Describe the "prior work"


### Opportunity {#lisus-assignment3-final-opportunity}

* What was wrong with the baseline / prior work / existing solution? Why did it need improvement?

Examples:
- there wasn't a previous implementation
- the previous performance, evaluated according to some specific metrics, was not satisfactory
- it was not robust / reliable
- somebody told me to do so (/s) (this is a terrible motivation. In general, never ever ever say "somebody told me to do it" or "everybody does like this")

* How did you go about improving the existing solution / approaching the problem? [contribution]

Examples:
- We used method / algorithm xyz to fix the gap in knowledge (don't go in the details here)
- Make sure to reference papers you used / took inspiration from, lessons, textbooks, third party projects and any other resource you took advantage of (check [here](+duckumentation#bibliography-support) how to add citations in this document). Even in your code, make sure you are giving credit in the comments to original authors if you are reusing some components.

## Background and Preliminaries {#lisus-assignment3-final-preliminaries}

* Is there some particular theorem / "mathy" thing you require your readers to know before delving in the actual problem? Briefly explain it and links for more detailed explanations here.

Definition of link:
- could be the reference to a paper / textbook
- (bonus points) it is best if it is a link to Duckiebook chapter (in the dedicated "Preliminaries" section)

## Definition of the problem {#lisus-assignment3-final-problem-def}

_Up to now it was all fun and giggles. This is the most important part of your report: a crisp, possibly mathematical, definition of the problem you tackled. You can use part of the preliminary design document to fill this section._

Make sure you include your:
- final objective / goal
- assumptions made
- quantitative performance metrics to judge the achievement of the goal

## Contribution / Added functionality {#lisus-assignment3-final-contribution}

Describe here, in technical detail, what you have done. Make sure you include:
- a theoretical description of the algorithm(s) you implemented
- logical architecture
- software architecture
- details on the actual implementation where relevant (how does the implementation differ from the theory?)
- any infrastructure you had to develop in order to implement your algorithm
- If you have collected a number of logs, add link to where you stored them

_Feel free to create subsections when useful to ease the flow_

## Formal performance evaluation / Results {#lisus-assignment3-final-formal}

_Be rigorous!_

- For each of the tasks you defined in you problem formulation, provide quantitative results (i.e., the evaluation of the previously introduced performance metrics)
- Compare your results to the success targets. Explain successes or failures.
- Compare your results to the "state of the art" / previous implementation where relevant. Explain failure / success.
- Include an explanation / discussion of the results. Where things (as / better than / worst than) you expected? What were the biggest challenges?

## Future avenues of development {#lisus-assignment3-final-next-steps}

_Is there something you think still needs to be done or could be improved? List it here, and be specific!_
