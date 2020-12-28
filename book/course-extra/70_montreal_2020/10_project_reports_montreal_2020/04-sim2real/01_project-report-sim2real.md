#  Group name: Project report {#groupname-final-report status=ready}

Before starting, you should have a look at some tips on [how to write beautiful Duckiebook pages](+duckumentation#book).

<!--
General notes:
- REMEMBER to change:
- "groupname" in the chapter labels to your group label (e.g. proj-xyz) - not doing this will cause the build to crash.
- "group name" every where you read it to a string describing your project; e.g., parking, self-calibration, learning to follow the lane, etc.
-->

_The objective of this report is to bring justice to your  hard work during the semester and make so that future generations of Duckietown students may take full advantage of it. Some of the sections of this report are repetitions from the preliminary design document (PDD) and intermediate report you have given._

## The final result {#sim2real-final-result}

_Let's start from a teaser._

* Post a video of your best results (e.g., your demo video): remember to have duckies on the robots or something terrible might happen!

You might want to add as a caption a link to your [instructions to reproduce](#instructions-sim2real) to reproduce these results.
Moreover, add a link to the readme.txt of your code.

## Mission and Scope {#sim2real-final-scope}

_Now tell your story:_

The mission of this project was to find a way to close the reality gap by using Generative Adversarial Networks 
or any other deep learning technique, to generate realistic images from simulated images. The idea was that by 
replacing the simulated image by a realistic image based on input image, we could train an agent in the simulator that would not have to be tuned seperately on the simulator and on the robot.


### Motivation {#sim2real-final-result-motivation}

_Now step back and tell us how you got to that mission._

- What are we talking about? [Brief introduction / problem in general terms]

- Why is it important? [Relevance]



### Existing solution {#sim2real-final-literature}

-  Describe the "prior work"


### Opportunity {#sim2real-final-opportunity}

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

## Background and Preliminaries {#sim2real-final-preliminaries}

* Is there some particular theorem / "mathy" thing you require your readers to know before delving in the actual problem? Briefly explain it and links for more detailed explanations here.

Definition of link:
- could be the reference to a paper / textbook
- (bonus points) it is best if it is a link to Duckiebook chapter (in the dedicated "Preliminaries" section)

## Definition of the problem {#sim2real-final-problem-def}

_Up to now it was all fun and giggles. This is the most important part of your report: a crisp, possibly mathematical, definition of the problem you tackled. You can use part of the preliminary design document to fill this section._

Make sure you include your:
- final objective / goal
- assumptions made
- quantitative performance metrics to judge the achievement of the goal

## Contribution / Added functionality {#sim2real-final-contribution}

Describe here, in technical detail, what you have done. Make sure you include:
- a theoretical description of the algorithm(s) you implemented
- logical architecture
- software architecture
- details on the actual implementation where relevant (how does the implementation differ from the theory?)
- any infrastructure you had to develop in order to implement your algorithm
- If you have collected a number of logs, add link to where you stored them

_Feel free to create subsections when useful to ease the flow_

## Formal performance evaluation / Results {#sim2real-final-formal}

_Be rigorous!_

- For each of the tasks you defined in you problem formulation, provide quantitative results (i.e., the evaluation of the previously introduced performance metrics)
- Compare your results to the success targets. Explain successes or failures.
- Compare your results to the "state of the art" / previous implementation where relevant. Explain failure / success.
- Include an explanation / discussion of the results. Where things (as / better than / worst than) you expected? What were the biggest challenges?

## Future avenues of development {#sim2real-final-next-steps}

_Is there something you think still needs to be done or could be improved? List it here, and be specific!_
