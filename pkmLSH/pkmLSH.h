/*
 Copyright (C) 2011 Parag K. Mital
 
 The Software is and remains the property of Parag K Mital
 ("pkmital") The Licensee will ensure that the Copyright Notice set
 out above appears prominently wherever the Software is used.
 
 The Software is distributed under this Licence:
 
 - on a non-exclusive basis,
 
 - solely for non-commercial use in the hope that it will be useful,
 
 - "AS-IS" and in order for the benefit of its educational and research
 purposes, pkmital makes clear that no condition is made or to be
 implied, nor is any representation or warranty given or to be
 implied, as to (i) the quality, accuracy or reliability of the
 Software; (ii) the suitability of the Software for any particular
 use or for use under any specific conditions; and (iii) whether use
 of the Software will infringe third-party rights.
 
 pkmital disclaims:
 
 - all responsibility for the use which is made of the Software; and
 
 - any liability for the outcomes arising from using the Software.
 
 The Licensee may make public, results or data obtained from, dependent
 on or arising out of the use of the Software provided that any such
 publication includes a prominent statement identifying the Software as
 the source of the results or the data, including the Copyright Notice
 and stating that the Software has been made available for use by the
 Licensee under licence from pkmital and the Licensee provides a copy of
 any such publication to pkmital.
 
 The Licensee agrees to indemnify pkmital and hold them
 harmless from and against any and all claims, damages and liabilities
 asserted by third parties (including claims for negligence) which
 arise directly or indirectly from the use of the Software or any
 derivative of it or the sale of any products based on the
 Software. The Licensee undertakes to make no liability claim against
 any employee, student, agent or appointee of pkmital, in connection
 with this Licence or the Software.
 
 
 No part of the Software may be reproduced, modified, transmitted or
 transferred in any form or by any means, electronic or mechanical,
 without the express permission of pkmital. pkmital's permission is not
 required if the said reproduction, modification, transmission or
 transference is done without financial return, the conditions of this
 Licence are imposed upon the receiver of the product, and all original
 and amended source code is included in any transmitted product. You
 may be held legally responsible for any copyright infringement that is
 caused or encouraged by your failure to abide by these terms and
 conditions.
 
 You are not permitted under this Licence to use this Software
 commercially. Use for which any financial return is received shall be
 defined as commercial use, and includes (1) integration of all or part
 of the source code or the Software into a product for sale or license
 by or on behalf of Licensee to third parties or (2) use of the
 Software or any derivative of it for research with the final aim of
 developing software products for sale or license to a third party or
 (3) use of the Software or any derivative of it for research with the
 final aim of developing non-software products for sale or license to a
 third party, or (4) use of the Software to provide any service to an
 external organisation for which payment is received. If you are
 interested in using the Software commercially, please contact pkmital to
 negotiate a licence. Contact details are: parag@pkmital.com
 */

/*
 Original LSH implementation from:
 
 Copyright (c) 2013, josiahw. All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:
 
 Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 */

#pragma once

#include <iostream>
#include <stdint.h>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <cmath>
#include <unordered_map>
#include <armadillo>

using namespace std;
using namespace arma;

namespace pkm
{
    template <class T>
    class LSH
    {
    public:
        
        /*!
         Constructor initialized from the number of tables and bits for each table
         @param n_tables            How many tables to construct
         @param n_bits_per_table    Number of bits to use for each table
         */
        LSH(uint32_t n_tables,
            uint32_t n_bits_per_table = 32)
        :
        n_total_hash_tables(n_tables),
        n_total_bits(n_bits_per_table),
        map_hashes(n_tables),
        b_initialized(false)
        {
            
        }
        
        ~LSH()
        {
            
        }
        
        
        /*!
         Initialze using a dataset
         @param mat_data - The dataset to encode
         */
        void initialize(const arma::Mat<T>& mat_data)
        {
            const size_t n_buckets = 65535;
            for (auto& map : map_hashes)
            {
                // Disable the load factor check on the hash tables
                map.max_load_factor(numeric_limits<float>::max());
                
                // Set the number of buckets
                map.rehash(n_buckets);
            }
            
            // Should only be called on instantiation
            buildHashes(mat_data);
            original_data.reserve(mat_data.n_rows);
            for (size_t row_i = 0; row_i < mat_data.n_rows; ++row_i)
            {
                insert(mat_data.row(row_i));
            }
            
            b_initialized = true;
        }
        
        bool isBuilt() {
            return b_initialized;
        }
        
        
        /*!
         Use LSH to find nearest indices to p
         @param p - The data to query
         */
        vector<size_t> knn(const arma::Row<T>& p)
        {
            const vector<size_t> neighbors = query(p);
            vector<size_t> range(neighbors.size());
            vector<size_t> new_neighbors(neighbors.size());
            vector<T> dists(neighbors.size());
            for (size_t neighbor_i = 0; neighbor_i < neighbors.size(); ++neighbor_i)
            {
                range[neighbor_i] = neighbor_i;
                const Row<T> diff = p - original_data[neighbors[neighbor_i]];
                dists[neighbor_i] = -dot(diff,diff);
            }
            sortIndices(dists, range);
            for (size_t neighbor_i = 0; neighbor_i < neighbors.size(); ++neighbor_i)
            {
                new_neighbors[neighbor_i] = neighbors[range[neighbor_i]];
            }
            return new_neighbors;
        }
        
        /*!
         Use LSH to find nearest indices to p
         @param p - The data to query
         */
        void knn(const arma::Row<T>& p, vector<T>& dists, vector<size_t>& nearest_neighbors)
        {
            const vector<size_t> neighbors = query(p);
            vector<size_t> range(neighbors.size());
            
            nearest_neighbors.resize(neighbors.size());
            dists.resize(neighbors.size());
            
            for (size_t neighbor_i = 0; neighbor_i < neighbors.size(); ++neighbor_i)
            {
                range[neighbor_i] = neighbor_i;
                const Row<T> diff = p - original_data[neighbors[neighbor_i]];
                dists[neighbor_i] = -dot(diff,diff);
            }
            sortIndices(dists, range);
            for (size_t neighbor_i = 0; neighbor_i < neighbors.size(); ++neighbor_i)
            {
                nearest_neighbors[neighbor_i] = neighbors[range[neighbor_i]];
            }
        }
        
        
        void insert(const Row<T>& data)
        {
            const vector<uint64_t> hash_values = getHashes(data);
            for (size_t table_i = 0; table_i < this->n_total_hash_tables; ++table_i) {
                map_hashes[table_i].insert(make_pair(hash_values[table_i], original_data.size()));
            }
            original_data.push_back(data);
        }
        
        
    protected:
        
        
        /*!
         Takes a dataset and organizes the hashes for LSH
         @param mat_data - The dataset to encode
         */
        void buildHashes(const arma::Mat<T>& mat_data)
        {
            // per-dimension means
            means = mean(mat_data);
            
            // per-dimension variances
            arma::Row<T> variances = var(mat_data);
            
            // variance sums
            T vSum = sum(variances);
            
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dist(0, vSum);
            hashes.clear();
            vector<uint8_t> used(mat_data.n_cols, 0);
            
            for (size_t table_i = 0; table_i < this->n_total_hash_tables; ++table_i)
            {
                vector<size_t> hash_fn_i;
                while (hash_fn_i.size() < this->n_total_bits)
                {
                    double sum_dist = variances[0];
                    double target = dist(gen);
                    size_t selection = 0;
                    while (sum_dist < target)
                    {
                        ++selection;
                        sum_dist += variances[selection];
                    }
                    bool repeated = (used[selection] == 2);
                    if (not repeated)
                    {
                        used[selection] = 2;
                        hash_fn_i.push_back(selection);
                    }
                }
                for(auto j : hash_fn_i)
                {
                    used[j] = 1;
                }
                hashes.push_back(move(hash_fn_i));
            }
            
            // Calculate the candidates to mean-split when we hash.
            for(size_t i = 0; i < used.size(); ++i)
            {
                if (used[i] == 1)
                {
                    prehash.push_back(i);
                }
            }
        }
        
        
        vector<uint64_t> getHashes(const Row<T>& p)
        {
            vector<uint64_t> tmp(p.n_cols);
            vector<uint64_t> result(n_total_hash_tables, 0);
            for (auto hash_i : prehash)
            {
                tmp[hash_i] = p[hash_i] > means[hash_i];
            }
            for (size_t table_i = 0; table_i < n_total_hash_tables; ++table_i)
            {
                for (auto hash_i : hashes[table_i])
                {
                    result[table_i] = (result[table_i] << 1) | tmp[hash_i];
                }
            }
            return result;
        }
        
        
        
        
        /*!
         Query for all the items in buckets and return a unique list of ID's
         */
        vector<size_t> query(const Row<T>& p)
        {
            const vector<uint64_t> hash_values = getHashes(p);
            vector<size_t> values;
            set<uint64_t> unique_values;
            
            // Get a list of bucket iterators. We know from the way we construct our table that ID's strictly increase.
            // Therefore we have a sorted list of ID's in each bucket. Sort through for unique ones.
            for (size_t table_i = 0; table_i < this->n_total_hash_tables; ++table_i)
            {
                const size_t bucketID = map_hashes[table_i].bucket(hash_values[table_i]);
                for (auto hash_i = map_hashes[table_i].begin(bucketID); hash_i != map_hashes[table_i].end(bucketID); ++hash_i)
                {
                    unique_values.insert(hash_i->second);
                }
            }
            
            values.reserve(unique_values.size());
            for (auto j = unique_values.begin(); j != unique_values.end(); ++j)
            {
                values.push_back((*j));
            }
            return values;
        }
        
        
    private:
        unsigned int n_total_hash_tables;                       //!< how many different hash tables
        unsigned int n_total_bits;                              //!< number of hashes per table
        
        vector<vector<size_t> > hashes;
        vector<arma::Row<T> > original_data;
        arma::Row<T> means;
        vector<unordered_map<uint64_t,size_t> > map_hashes;
        vector<size_t> prehash;
        
        bool b_initialized;                                     //!< has the database been initialized with data
        
    private:
        void sortIndices(const vector<T>& sortBy, vector<size_t>& indices) {
            for (size_t idx_i = 0; idx_i < indices.size(); ++idx_i) {
                indices[idx_i] = idx_i;
            }
            
            // get sorted indices
            sort(begin(indices),end(indices),
                 [&sortBy](const size_t& a, const size_t& b) {
                     return sortBy[a] > sortBy[b];
                 });
        }
        
    };
    
};
