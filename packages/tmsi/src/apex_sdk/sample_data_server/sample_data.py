'''
(c) 2022 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file sample_data.py 
 * @brief 
 * Sample data structure.
 */


'''

class SampleSet:
    def __init__(self, num_samples, samples):
        self.num_samples = num_samples
        self.samples = samples

class SampleData:
    def __init__(self, num_sample_sets, num_samples_per_sample_set, samples):
        self.num_sample_sets = num_sample_sets
        self.num_samples_per_sample_set = num_samples_per_sample_set
        self.samples = samples

class SampleDataConsumer:
    def __init__(self, id, q):
        self.id = id
        self.q = q