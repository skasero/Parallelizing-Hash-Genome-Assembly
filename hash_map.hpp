#pragma once

#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"

using namespace std;

struct HashMap {
    std::vector<upcxx::global_ptr<kmer_pair>> data; // Global vector pointer of kmer_pairs
    std::vector<upcxx::global_ptr<int>> used; // Global vector pointer of used slots 

	size_t total_size; // Total size of entire problem
    size_t my_size; // Current size for each rank
    size_t n_proc; // Total number of processors
	size_t rank; // Current rank 
    size_t size() const noexcept;

	upcxx::atomic_domain<int> ad; // Atomic used for request_slot(). Must be declared here as I couldn't get it to work with using a destory()

    HashMap(size_t size);
	~HashMap();

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair & kmer);
    bool find(const pkmer_t & key_kmer, kmer_pair & val_kmer);

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair & kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

// The atomic must be declared here
HashMap::HashMap(size_t size): ad({upcxx::atomic_op::compare_exchange}) {
	n_proc = upcxx::rank_n();
	rank = upcxx::rank_me();
    total_size = size;
	my_size = (size + n_proc - 1) / n_proc;

	data.resize(n_proc);
	used.resize(n_proc);

	data[rank] = upcxx::new_array<kmer_pair>(my_size);
	used[rank] = upcxx::new_array<int>(my_size);

	for(int i = 0; i < n_proc; i++){
		data[i] = upcxx::broadcast(data[i],i).wait();
		used[i] = upcxx::broadcast(used[i],i).wait();
	}
}

HashMap::~HashMap(){
	upcxx::delete_array(data[rank]);
	upcxx::delete_array(used[rank]);
}

bool HashMap::insert(const kmer_pair & kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;

    do {
        uint64_t slot = (hash + probe++) % size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::find(const pkmer_t & key_kmer, kmer_pair & val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) {
	size_t rank_number = slot / my_size;
	size_t offset = slot % my_size;
    return upcxx::rget(used[rank_number] + offset).wait();
}

void HashMap::write_slot(uint64_t slot, const kmer_pair & kmer) {
    size_t rank_number = slot / my_size;
	size_t offset = slot % my_size;
    upcxx::rput(kmer, data[rank_number] + offset).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    size_t rank_number = slot / my_size;
	size_t offset = slot % my_size;
    return upcxx::rget(data[rank_number] + offset).wait();
}

bool HashMap::request_slot(uint64_t slot) {
	size_t rank_number = slot / my_size;
	size_t offset = slot % my_size;

	// compare_exchange first checks if the value is set to 0. If so, it sets the value to 1.
	bool val = ad.compare_exchange(used[rank_number] + offset, 0, 1, std::memory_order_relaxed).wait();
	return !val;
}

size_t HashMap::size() const noexcept {
    return total_size;
}
