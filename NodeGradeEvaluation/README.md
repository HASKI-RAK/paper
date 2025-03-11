# Evaluation of a Node-based Automatic Short Answer Tool “NodeGrade”
NodeGrade tries to provide a suitable solution for the problem of time-intensive short answer grading. This research focuses simultaneously on performance, functionality and user experience, which is underlined by a triangulated approach. The evaluation results show comparable performance of NodeGrade on public datasets, even outperforming GPT-4 on the SemEval 2013 Task 7. Matching of NodeGrade's output with multiple human expert raters reveals some weaknesses regarding cases at the lower and upper boundary. In terms of user experience, the interviewed and observed students recognized both positive facets, like better learning support and helpful feedback, and negative sides, including technical limitations and a lack of transparency, while also suggesting future improvements. Overall, NodeGrade promises high potential for further practical use and testing in the field of software engineering education and automatic short answer grading.

## Structure
The repository is structured as follows:
```
/
├── benchmark/                 # Code to download the datasets and run the evaluation of NodeGrade
├── interviews/                # Qualitative interviews with students
```

### Benchmark
In the folder `benchmark/` you will find the code to download the datasets and run the evaluation of NodeGrade.
To setup NodeGrade, plase install the application according to the repository [NodeGrade](https://github.com/HASKI-RAK/NodeGrade).
The graph confugrations used are stored in the `benchmark/graphs/` folder. Upload them in the web interface of NodeGrade.
Refer to the [README](benchmark/README.md) in the `benchmark/` folder for more information.

### Interviews
In the folder `interviews/` you will find the qualitative interviews with students.

## Issues
If you have any questions or issues, please open an issue in the repository. We will try to help you as soon as possible.

## Citation
If you use this code or the datasets in your research, please cite the following paper:
```
...
```