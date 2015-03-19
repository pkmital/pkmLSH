/*
 
 Testing and class interface for Locality Sensitive Hashing 
 
 by Parag K. Mital
 
 Copyright 2015 Parag K. Mital, http://pkmital.com
 
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

#include "pkmLSH.h"
#include <armadillo>

using namespace pkm;

void test()
{
    // Create dataset
    uword n_observations = 200000;
    uword n_features = 50;
    mat data = randn(n_observations, n_features);
    cout << "Created random dataset" << endl;
    
    
    // Create LSH
    uint32_t n_tables = log(log(n_observations));
    uint32_t n_bits = 32;
    auto start = chrono::steady_clock::now();
    LSH lsh_table(n_tables, n_bits);
    lsh_table.initialize(data);
    auto end = chrono::steady_clock::now();
    cout << "Created LSH table with " << n_tables << " tables and " << n_bits << " bits per table in " << double((end-start).count())/double(chrono::steady_clock::period::den) << " seconds" << endl;
    
    // Test it by querying every row
    size_t n_correct = 0;
    double avg_time = 0;
    for (auto row_i = 0; row_i < n_observations; row_i++)
    {
        auto start = chrono::steady_clock::now();
        auto idx = lsh_table.knn(data.row(row_i));
        auto end = chrono::steady_clock::now();
        if (idx[0] == row_i) {
            n_correct++;
        }
        avg_time += double((end-start).count())/double(chrono::steady_clock::period::den);
    }
    
    // Report results
    cout << n_correct << "/" << n_observations << " (" << n_correct / (double)n_observations * 100.0 << " %) correct using " << avg_time / (double)n_observations << " seconds per observation" << endl;
}


int main(int argc, const char * argv[]) {
    
    test();
    
    return 0;
}
